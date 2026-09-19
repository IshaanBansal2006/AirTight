"""Replay an episode log: script the person, mark drones, dispatch at alarm time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from airtight.contracts.episode import (
    AlarmEvent,
    OutcomeEvent,
    PositionEvent,
    read_episode_log,
)
from airtight.dimos_lane.modules.orchestrator import Orchestrator
from airtight.dimos_lane.person import interpolate_path

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.contracts.site import XY


@dataclass
class ReplayPlan:
    title: str
    seed: int
    tactic_id: str
    timely_detected: bool
    t_alarm: float | None
    t_cdp: float
    intruder: list[tuple[float, XY]]
    markers: dict[str, list[tuple[float, XY]]]
    dispatch_at: float | None
    dispatch_xy: XY | None
    dispatch_result: str | None = None


def _positions_by_object(
    events: list[object],
) -> dict[str, list[tuple[float, XY]]]:
    out: dict[str, list[tuple[float, XY]]] = {}
    for event in events:
        if isinstance(event, PositionEvent):
            out.setdefault(event.object_id, []).append((event.t, event.position))
    return out


def plan_replay(path: Path, *, dispatch: bool = True) -> ReplayPlan:
    header, event_iter = read_episode_log(path)
    events = list(event_iter)
    positions = _positions_by_object(events)
    outcome = next(e for e in events if isinstance(e, OutcomeEvent))
    alarm = next((e for e in events if isinstance(e, AlarmEvent)), None)
    intruder = positions.get("intruder", [])
    markers = {oid: pts for oid, pts in positions.items() if oid != "intruder"}
    dispatch_xy = intruder[-1][1] if intruder else None
    if alarm is not None and intruder:
        earlier = [p for p in intruder if p[0] <= alarm.t]
        if earlier:
            dispatch_xy = earlier[-1][1]
    title = "catch" if outcome.timely_detected else "miss"
    plan = ReplayPlan(
        title=title,
        seed=header.seed,
        tactic_id=header.tactic.id,
        timely_detected=outcome.timely_detected,
        t_alarm=outcome.t_alarm,
        t_cdp=outcome.t_cdp,
        intruder=intruder,
        markers=markers,
        dispatch_at=alarm.t if alarm is not None else None,
        dispatch_xy=dispatch_xy,
    )
    if dispatch and dispatch_xy is not None:
        orch = Orchestrator()
        plan.dispatch_result = orch.dispatch_verify(dispatch_xy.x, dispatch_xy.y)
    return plan


def resample_intruder(plan: ReplayPlan, speed_mps: float, dt: float) -> list[tuple[float, XY]]:
    if len(plan.intruder) < 2:
        return list(plan.intruder)
    waypoints = [p for _, p in plan.intruder]
    return interpolate_path(waypoints, speed_mps=speed_mps, dt=dt)


def apply_person_poses(samples: list[tuple[float, XY]], *, live: bool = False) -> int:
    """Optionally publish each sample on /person_pose. Returns the count."""
    if not live:
        return len(samples)
    from math import atan2

    from airtight.dimos_lane.person import publish_person_pose

    prev = samples[0][1]
    for _, xy in samples:
        yaw = atan2(xy.y - prev.y, xy.x - prev.x)
        publish_person_pose(xy.x, xy.y, yaw_rad=yaw)
        prev = xy
    return len(samples)
