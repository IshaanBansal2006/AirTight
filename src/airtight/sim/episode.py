"""The v0 episode: a patrolling fleet and fixed sensors against one intruder and benign traffic.

One clock. Step k has t = (k - n_warm) * dt, computed from k every step and never accumulated,
so t = 0 (the intruder on its entry point) is hit exactly. Warm-up runs at negative t: agents
patrol, nobody looks, nothing is scored or recorded.

Each step, in order: mark cells seen, retarget, then for t >= 0 the looks, then the recorder,
then the agents move. Looks therefore use positions at time t, before anyone moves.

The result is threshold-free. The alarm threshold is applied offline, so EpisodeScores carries
peaks, not a verdict.

task_time_s is a lane-local what-if. The contract's asset has no task time, so the intruder
wins the moment it arrives and t_cdp = t_reach - response_time_s. With task_time_s > 0 the
intruder must also stay on the asset that long, so inside simulate only
t_cdp = max(t_reach + task_time_s - response_time_s, 0) and the episode runs that much longer.
It exists to show the team what a task-time field on the asset would do. run_episode never sets
it, so the contract result is unchanged, and it goes away the day the contract has the field.

v0 ignores: endurance and charging, the tactic's phase, comms mode and comms events, tasks, and
any reaction to the decoy (it is scored like an intruder, nobody is sent to it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from airtight.sim import adapt
from airtight.sim.actors import Intruder, make_decoy, spawn_benign
from airtight.sim.constants import DEFAULT_CELL_SIZE_M, TAU_REF, TIME_EPS
from airtight.sim.fleet import PatrolController, make_agents, step_agents
from airtight.sim.geometry import Grid, inside_mask, patrol_weight
from airtight.sim.sensing import (
    LookSchedule,
    ScoreBook,
    do_looks,
    make_fixed_observers,
    sensor_rng,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    import numpy.typing as npt

    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
    from airtight.sim.actors import SimObject
    from airtight.sim.sensing import Look, Observer

    Array = npt.NDArray[np.float64]


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


def simulate(
    site: Site,
    fleet: FleetConfig,
    tactic: Tactic,
    sensor_curves: SensorCurves,
    seed: int,
    params: EpisodeParams = EpisodeParams(),  # noqa: B008  frozen, so a shared default is safe
    recorder: Recorder | None = None,
) -> EpisodeScores:
    check_setup(site, fleet, sensor_curves, params)
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

    intruder = Intruder(site, tactic)
    decoy = make_decoy(tactic)
    t_cdp = max(intruder.t_reach + params.task_time_s - adapt.response_time_s(site), 0.0)
    t_end = intruder.t_reach + params.task_time_s + params.tail_s
    benign = spawn_benign(site, 0.0, t_end, seed)
    objects: list[SimObject] = [intruder, *([decoy] if decoy is not None else []), *benign]

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
    rngs = {o.agent_id: sensor_rng(seed, o.agent_id) for o in observers}
    book = ScoreBook()
    n_looks = 0

    for k in range(n_warm + n_run + 1):
        t = (k - n_warm) * dt
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
        step_agents(agents, dt)

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
