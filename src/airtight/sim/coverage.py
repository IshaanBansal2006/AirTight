"""When the fleet is up, and how stale the yard gets while it is not.

Two views of the same thing. duty_intervals and uncovered_intervals are arithmetic, straight
from the battery clocks: exact, instant, and what tells the red team where to look. The
coverage profile is simulated, fleet only, and also counts the time an agent spends flying home,
when it still senses but no longer patrols.

Everything is laid out over one reference cycle, adapt.reference_cycle_s(fleet), the span that
Tactic.phase is a fraction of, so an interval here maps directly onto a range of phases. For a
fleet whose agents share one cycle that span is the cycle. For a mixed fleet it is the team's
mean cycle, which is not any agent's period; the intervals are still exact for that span.

An agent that never charges is always on duty. A fleet with one therefore has no uncovered
interval, which matches the simulated profile, where that agent is always patrolling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from airtight.sim import adapt
from airtight.sim.battery import make_clocks
from airtight.sim.episode import _run_loop, check_setup, official_params

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import numpy.typing as npt

    from airtight.contracts import FleetConfig, SensorCurves, Site
    from airtight.sim.episode import EpisodeParams
    from airtight.sim.fleet import AgentState, PatrolController

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]

SAMPLE_S = 1.0
_EPS = 1e-9


class UncoveredInterval(NamedTuple):
    start_s: float  # absolute seconds within the reference cycle
    end_s: float
    start_phase: float  # the same two numbers as fractions of the reference cycle
    end_phase: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class CoverageProfile(NamedTuple):
    t_s: Array  # sample times, one per SAMPLE_S, over [0, duration_s)
    on_duty: IntArray  # agents not charging (patrolling or flying home)
    patrolling: IntArray  # agents taking patrol targets
    max_staleness_s: Array  # over cells with positive patrol weight
    mean_staleness_s: Array
    duration_s: float


def duty_intervals(fleet: FleetConfig) -> dict[str, list[tuple[float, float]]]:
    """For each agent that charges, its on-duty intervals within [0, reference cycle)."""
    span = adapt.reference_cycle_s(fleet)
    out: dict[str, list[tuple[float, float]]] = {}
    for agent_id, clock in make_clocks(fleet).items():
        intervals = []
        first = math.floor(clock.offset_s / clock.cycle_s) - 1
        last = math.ceil((span + clock.offset_s) / clock.cycle_s)
        for k in range(first, last + 1):
            start = k * clock.cycle_s - clock.offset_s  # on duty while 0 <= u < endurance_s
            lo, hi = max(start, 0.0), min(start + clock.endurance_s, span)
            if hi - lo > _EPS:
                intervals.append((lo, hi))
        out[agent_id] = intervals
    return out


def uncovered_intervals(
    fleet: FleetConfig, only: Collection[str] | None = None
) -> list[UncoveredInterval]:
    """The stretches of the reference cycle in which no agent is on duty, in time order.

    only, if given, restricts the question to those agent ids: the stretches in which none of
    THEM is on duty, still laid out over the whole fleet's reference cycle so the phases line up
    with Tactic.phase. It answers "when are all the drones down" for a fleet that also has a
    guard, where the unrestricted answer is "never".

    Intervals are not joined across the end of the cycle: a gap that runs through it appears as
    one interval ending at the cycle and one starting at 0.
    """
    span = adapt.reference_cycle_s(fleet)
    duty = duty_intervals(fleet)
    considered = list(adapt.agent_ids(fleet)) if only is None else [a for a in only]
    unknown = set(considered) - set(adapt.agent_ids(fleet))
    if unknown:
        raise ValueError(f"agents {sorted(unknown)} are not in fleet {adapt.agent_ids(fleet)}")
    if not considered:
        return []
    if any(agent_id not in duty for agent_id in considered):
        return []  # one of them never charges, so somebody is always on duty
    duty = {agent_id: duty[agent_id] for agent_id in considered}
    gaps: list[UncoveredInterval] = []
    cursor = 0.0
    for lo, hi in sorted(iv for intervals in duty.values() for iv in intervals):
        if lo - cursor > _EPS:
            gaps.append(UncoveredInterval(cursor, lo, cursor / span, lo / span))
        cursor = max(cursor, hi)
    if span - cursor > _EPS:
        gaps.append(UncoveredInterval(cursor, span, cursor / span, 1.0))
    return gaps


def uncovered_s_per_hour_exact(fleet: FleetConfig) -> float:
    """The arithmetic value: seconds per hour with no agent on duty."""
    total = sum(gap.duration_s for gap in uncovered_intervals(fleet))
    return total / adapt.reference_cycle_s(fleet) * 3600.0


def coverage_profile(
    site: Site,
    fleet: FleetConfig,
    sensor_curves: SensorCurves,
    params: EpisodeParams | None = None,
    seed: int = 0,
) -> CoverageProfile:
    """Simulate the fleet alone over one reference cycle from absolute time 0.

    No objects, no looks, no start jitter: the clock is lined up with duty_intervals. It shares
    the episode loop, so the patrol here is the patrol an episode sees. params defaults to
    official_params().
    """
    params = official_params() if params is None else params
    check_setup(site, fleet, sensor_curves, params)
    duration = adapt.reference_cycle_s(fleet)
    rows: list[tuple[float, int, int, float, float]] = []

    def probe(t: float, agents: Sequence[AgentState], controller: PatrolController) -> None:
        if abs(t - round(t / SAMPLE_S) * SAMPLE_S) > _EPS or t > duration - _EPS:
            return
        stale = controller.staleness(t)[controller.weight > 0]
        rows.append(
            (
                t,
                sum(a.active for a in agents),
                sum(a.patrolling for a in agents),
                float(stale.max()) if stale.size else 0.0,
                float(stale.mean()) if stale.size else 0.0,
            )
        )

    _run_loop(site, fleet, sensor_curves, seed, params, [], duration, 0.0, probe=probe)
    table = np.array(rows, dtype=np.float64)
    return CoverageProfile(
        t_s=table[:, 0],
        on_duty=table[:, 1].astype(np.int64),
        patrolling=table[:, 2].astype(np.int64),
        max_staleness_s=table[:, 3],
        mean_staleness_s=table[:, 4],
        duration_s=duration,
    )


def uncovered_s_per_hour(profile: CoverageProfile) -> float:
    """Seconds per hour with zero agents patrolling. Fills Report.coverage_gap_s_per_hour.

    Larger than the arithmetic value by the time agents spend flying home, when they still
    sense but no longer take patrol targets.
    """
    uncovered_s = float(np.count_nonzero(profile.patrolling == 0)) * SAMPLE_S
    return uncovered_s / profile.duration_s * 3600.0
