"""The v0 episode: a patrolling fleet and fixed sensors against one intruder and benign traffic.

One clock. Step k has t = (k - n_warm) * dt, computed from k every step and never accumulated,
so t = 0 (the intruder on its entry point) is hit exactly. Warm-up runs at negative t: agents
patrol, nobody looks, nothing is scored or recorded.

Each step, in order: mark cells seen, retarget, then for t >= 0 the looks, then the recorder,
then the agents move. Looks therefore use positions at time t, before anyone moves.

Benign traffic is drawn over the fixed window [0, benign_window_s], never over [0, t_end]. The
end of the episode depends on the tactic, so a window tied to it would give two tactics
different animals under the same seed. With a fixed window the benign world is identical for
every tactic and every fleet under one seed.

The result is threshold-free. The alarm threshold is applied offline, so EpisodeScores carries
peaks, not a verdict.

task_time_s is a lane-local what-if. The contract's asset has no task time, so the intruder
wins the moment it arrives and t_cdp = t_reach - response_time_s. With task_time_s > 0 the
intruder must also stay on the asset that long, so inside simulate only
t_cdp = max(t_reach + task_time_s - response_time_s, 0) and the episode runs that much longer.
It exists to show the team what a task-time field on the asset would do. run_episode never sets
it, so the contract result is unchanged, and it goes away the day the contract has the field.

Battery and phase. With EpisodeParams.battery on, each agent that charges follows its battery
clock (battery.py) and the episode is placed in absolute time:
t0_abs = phase * reference_cycle_s + jitter, where reference_cycle_s is the team's definition of
"the fleet's charge cycle" (adapt.reference_cycle_s) and jitter is uniform on
[-phase_jitter_s, +phase_jitter_s] from the reserved intruder stream default_rng([seed, 2]). The
jitter is an honesty parameter: the adversary knows the schedule, not the second. It is always
drawn, battery on or off, so the stream never shifts. At the start of warm-up an agent on duty
starts on its pad and the warm-up disperses it, so an agent that is about to return docks a
little early, by at most one transit time.

run_episode turns the battery on. simulate leaves it off by default for now, so the part 1
sanity tables and tests keep their meaning: the yard_night tactics were written with phase 0.45
before phase meant anything, and that lands inside the charging window. The default flips once
the scenario's reference phase is chosen.

v0 still ignores: comms mode and comms events, tasks, battery log events, dock capacity, and any
reaction to the decoy (it is scored like an intruder, nobody is sent to it).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from airtight.sim import adapt
from airtight.sim.actors import INTRUDER_STREAM, Intruder, make_decoy, spawn_benign
from airtight.sim.battery import make_clocks, step_battery
from airtight.sim.constants import BENIGN_HORIZON_S, DEFAULT_CELL_SIZE_M, TAU_REF, TIME_EPS
from airtight.sim.fleet import PatrolController, make_agents, step_agents
from airtight.sim.geometry import WEIGHT_MODES, Grid, inside_mask, patrol_weight
from airtight.sim.sensing import (
    LookRngs,
    LookSchedule,
    ScoreBook,
    do_looks,
    make_fixed_observers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy.typing as npt

    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
    from airtight.sim.actors import SimObject
    from airtight.sim.fleet import AgentState
    from airtight.sim.sensing import Look, Observer

    Array = npt.NDArray[np.float64]
    Probe = Callable[[float, Sequence[AgentState], PatrolController], None]


@dataclass(frozen=True)
class EpisodeParams:
    dt: float = 0.25
    warmup_s: float = 150.0
    tail_s: float = 10.0
    cell_size_m: float = DEFAULT_CELL_SIZE_M
    retarget_period_s: float = 10.0
    d0_m: float = 60.0
    top_fraction: float = 0.05
    weight_base: float = 0.3
    weight_scale_m: float = 25.0
    asset_gain: float = 1.0
    weight_mode: str = "asset"  # one of geometry.WEIGHT_MODES
    v_ref_mps: float = 2.5  # the intruder speed the defender plans against; sets the band ring
    task_time_s: float = 0.0  # WHAT-IF only, see the module docstring; run_episode never sets it
    battery: bool = False  # follow the battery clocks; run_episode sets it, see the docstring
    phase_jitter_s: float = 15.0  # the adversary knows the schedule, not the second


@dataclass(frozen=True)
class EpisodeScores:
    """Lane B internal and threshold-free. The contract's EpisodeResult is derived from it."""

    seed: int
    intruder_peak: float  # at or before t_cdp; NEVER_SEEN if never looked at by then
    intruder_t_alarm_ref: float | None  # first crossing of TAU_REF at any time up to t_end
    benign_peaks: dict[str, float]  # every benign object ever looked at; never the decoy
    decoy_peak: float | None  # None when the tactic has no decoy
    sim_hours: float
    t_reach: float
    t_cdp: float
    t_end: float
    n_looks: int


@dataclass(frozen=True)
class QuietScores:
    """One quiet night: benign traffic only. Threshold-free, like EpisodeScores."""

    seed: int
    benign_peaks: dict[str, float]  # every benign object that was ever looked at
    sim_hours: float
    duration_s: float


class Recorder(Protocol):
    """What the loop tells the outside world, for t >= 0 only. It knows nothing of log formats."""

    def on_poses(self, t: float, poses: Mapping[str, Array]) -> None:
        """Moving agents and every alive object, keyed by id, at time t."""
        ...

    def on_looks(self, t: float, looks: Sequence[Look]) -> None: ...

    def on_finish(self, scores: EpisodeScores) -> None: ...


def check_setup(
    site: Site, fleet: FleetConfig, sensor_curves: SensorCurves, params: EpisodeParams
) -> None:
    """Raise one ValueError listing every problem, so nothing can fail mid-run."""
    problems: list[str] = []
    users = [(f"agent {a!r}", adapt.agent_sensor_type(fleet, a)) for a in adapt.agent_ids(fleet)]
    users += [(f"fixed sensor {s.sensor_id!r}", s.sensor_type) for s in adapt.fixed_sensors(site)]
    ids = [*adapt.agent_ids(fleet), *(s.sensor_id for s in adapt.fixed_sensors(site))]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"id {dup!r} is used by more than one agent or fixed sensor")
    in_use: list[str] = []
    for who, sensor_type in users:
        if not adapt.has_curve(sensor_curves, sensor_type):
            problems.append(f"{who} has sensor_type {sensor_type!r}, which has no sensor curve")
        elif sensor_type not in in_use:
            in_use.append(sensor_type)
    benign_classes = sorted({r.cls for r in adapt.benign_routes(site)})
    for sensor_type in in_use:
        known = adapt.fp_classes(sensor_curves, sensor_type)
        for cls in benign_classes:
            if cls not in known:
                problems.append(
                    f"sensor {sensor_type!r} has no false-positive rate for benign class {cls!r}"
                )
        period = 1.0 / adapt.sensor_look_rate_hz(sensor_curves, sensor_type)
        if period < params.dt - TIME_EPS:
            problems.append(
                f"sensor {sensor_type!r} looks every {period:.4f} s, shorter than dt {params.dt} s"
            )
    if problems:
        raise ValueError("episode setup is invalid:\n  - " + "\n  - ".join(problems))


WEIGHT_MODE_ENV = "AIRTIGHT_WEIGHT_MODE"


def official_params() -> EpisodeParams:
    """The parameters run_episode uses: the one source of truth for official numbers.

    The sweep calls this too, so a sweep can never disagree with run_episode. simulate's own
    default keeps the battery off so the part 1 tables keep their meaning.

    One documented override: the environment variable AIRTIGHT_WEIGHT_MODE sets the patrol
    weight mode (asset, uniform or band). Unset means asset. It exists because the weight mode
    is an engine parameter, not a fleet field, so a fix that changes it can only reach
    run_episode, and lane C's re-attack, this way. Nothing else can be overridden.
    """
    mode = os.environ.get(WEIGHT_MODE_ENV)
    if mode is None or mode == "":
        return EpisodeParams(battery=True)
    if mode not in WEIGHT_MODES:
        raise ValueError(
            f"{WEIGHT_MODE_ENV}={mode!r} is not a weight mode; choose one of {WEIGHT_MODES}"
        )
    return EpisodeParams(battery=True, weight_mode=mode)


def _start_jitter_s(seed: int, params: EpisodeParams) -> float:
    """Uniform on [-phase_jitter_s, +phase_jitter_s] from the reserved intruder stream."""
    rng = np.random.default_rng([seed, INTRUDER_STREAM])
    return float(rng.uniform(-params.phase_jitter_s, params.phase_jitter_s))


def _run_loop(
    site: Site,
    fleet: FleetConfig,
    sensor_curves: SensorCurves,
    seed: int,
    params: EpisodeParams,
    objects: Sequence[SimObject],
    t_end: float,
    t0_abs: float,
    recorder: Recorder | None = None,
    probe: Probe | None = None,
) -> tuple[ScoreBook, int]:
    """The one loop behind simulate, simulate_quiet and coverage_profile.

    Runs from the start of warm-up to t_end and returns the score book and the number of looks.
    t0_abs is the absolute time of t = 0, for the battery clocks. probe, if given, is called
    once per step for t >= 0 with (t, agents, controller), after the looks and before anyone
    moves.
    """
    dt = params.dt
    grid = Grid(*adapt.bounds(site), params.cell_size_m)
    weight = patrol_weight(
        grid,
        inside_mask(grid, adapt.perimeter(site)),
        adapt.assets(site),
        base=params.weight_base,
        scale_m=params.weight_scale_m,
        asset_gain=params.asset_gain,
        mode=params.weight_mode,
        r_c=adapt.critical_radius_m(site, params.v_ref_mps),
    )
    agents = make_agents(site, fleet, sensor_curves)
    observers: list[Observer] = [*agents, *make_fixed_observers(site, sensor_curves)]

    n_warm = math.ceil(params.warmup_s / dt - TIME_EPS)
    n_run = math.floor(t_end / dt + TIME_EPS)
    controller = PatrolController(
        grid,
        weight,
        seed,
        t_start=-n_warm * dt,
        retarget_period_s=params.retarget_period_s,
        d0_m=params.d0_m,
        top_fraction=params.top_fraction,
    )
    schedule = LookSchedule(
        {o.agent_id: adapt.sensor_look_rate_hz(sensor_curves, o.sensor_type) for o in observers},
        dt,
    )
    rngs = LookRngs(seed)
    clocks = make_clocks(fleet) if params.battery else {}
    book = ScoreBook()
    n_looks = 0

    for k in range(n_warm + n_run + 1):
        t = (k - n_warm) * dt
        if clocks:
            step_battery(agents, clocks, t0_abs + t, dt)
        controller.mark_seen(observers, t)
        controller.retarget(agents, t)
        if k >= n_warm:
            looks = do_looks(observers, objects, t, sensor_curves, rngs, schedule, book)
            n_looks += len(looks)
            if recorder is not None:
                poses = {a.agent_id: a.pos.copy() for a in agents}
                poses.update({o.object_id: o.position(t) for o in objects if o.alive(t)})
                recorder.on_poses(t, poses)
                recorder.on_looks(t, looks)
            if probe is not None:
                probe(t, agents, controller)
        step_agents(agents, dt)
    return book, n_looks


def simulate(
    site: Site,
    fleet: FleetConfig,
    tactic: Tactic,
    sensor_curves: SensorCurves,
    seed: int,
    params: EpisodeParams = EpisodeParams(),  # noqa: B008  frozen, so a shared default is safe
    recorder: Recorder | None = None,
    benign_window_s: float = BENIGN_HORIZON_S,
) -> EpisodeScores:
    check_setup(site, fleet, sensor_curves, params)

    intruder = Intruder(site, tactic)
    decoy = make_decoy(tactic)
    t_cdp = max(intruder.t_reach + params.task_time_s - adapt.response_time_s(site), 0.0)
    t_end = intruder.t_reach + params.task_time_s + params.tail_s
    if t_end > benign_window_s + TIME_EPS:
        raise ValueError(
            f"the episode ends at t_end = {t_end:.1f} s, after the benign window of "
            f"{benign_window_s:.1f} s; raise benign_window_s or shorten the tactic"
        )
    benign = spawn_benign(site, 0.0, benign_window_s, seed)
    objects: list[SimObject] = [intruder, *([decoy] if decoy is not None else []), *benign]
    t0_abs = adapt.tactic_phase(tactic) * adapt.reference_cycle_s(fleet) + _start_jitter_s(
        seed, params
    )

    book, n_looks = _run_loop(
        site, fleet, sensor_curves, seed, params, objects, t_end, t0_abs, recorder
    )

    benign_ids = {b.object_id for b in benign}
    scores = EpisodeScores(
        seed=seed,
        intruder_peak=book.peak(intruder.object_id, t_max=t_cdp),
        intruder_t_alarm_ref=book.first_crossing(intruder.object_id, TAU_REF),
        benign_peaks={oid: book.peak(oid) for oid in book.object_ids() if oid in benign_ids},
        decoy_peak=book.peak(decoy.object_id) if decoy is not None else None,
        sim_hours=t_end / 3600.0,
        t_reach=intruder.t_reach,
        t_cdp=t_cdp,
        t_end=t_end,
        n_looks=n_looks,
    )
    if recorder is not None:
        recorder.on_finish(scores)
    return scores


def simulate_quiet(
    site: Site,
    fleet: FleetConfig,
    sensor_curves: SensorCurves,
    seed: int,
    duration_s: float | None = None,
    params: EpisodeParams | None = None,
) -> QuietScores:
    """A quiet night: the fleet and benign traffic, no intruder and no decoy.

    This is where false alarm rates come from. An intrusion window is about a minute, far too
    little benign exposure to pin a rate of one per hour. duration_s defaults to the fleet's
    reference cycle, so one run samples every charge phase. Benign traffic is drawn over the
    whole duration. The run starts at phase 0 with the usual jitter draw. params defaults to
    official_params().
    """
    params = official_params() if params is None else params
    duration = adapt.reference_cycle_s(fleet) if duration_s is None else float(duration_s)
    if not duration > 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    check_setup(site, fleet, sensor_curves, params)
    benign = spawn_benign(site, 0.0, duration, seed)
    book, _ = _run_loop(
        site, fleet, sensor_curves, seed, params, benign, duration, _start_jitter_s(seed, params)
    )
    return QuietScores(
        seed=seed,
        benign_peaks={oid: book.peak(oid) for oid in book.object_ids()},
        sim_hours=duration / 3600.0,
        duration_s=duration,
    )
