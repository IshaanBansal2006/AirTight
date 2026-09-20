"""Replay an episode log: script the person, mark drones, dispatch at alarm time."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from airtight.contracts.episode import (
    AlarmEvent,
    OutcomeEvent,
    PositionEvent,
    read_episode_log,
)
from airtight.contracts.site import XY
from airtight.dimos_lane.modules.orchestrator import Orchestrator
from airtight.dimos_lane.person import interpolate_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


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
    speed_mps: float = 1.6
    waypoints: list[XY] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayTick:
    t: float
    person: XY | None
    markers: dict[str, XY]
    dispatch: bool


def _positions_by_object(
    events: Sequence[object],
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
    if not markers:
        markers = _default_marker_tracks()
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
        speed_mps=header.tactic.speed_mps,
        waypoints=list(header.tactic.waypoints),
    )
    if dispatch and dispatch_xy is not None:
        orch = Orchestrator()
        plan.dispatch_result = orch.dispatch_verify(dispatch_xy.x, dispatch_xy.y)
    return plan


def _default_marker_tracks() -> dict[str, list[tuple[float, XY]]]:
    """Kinematic fleet poses when the episode log has no agent tracks."""
    from airtight.dimos_lane.modules.sim_fleet import SimFleet

    fleet = SimFleet()
    return {
        did: [(0.0, XY(x=float(pos[0]), y=float(pos[1])))] for did, pos in fleet.snapshot().items()
    }


def iter_ticks(plan: ReplayPlan) -> list[ReplayTick]:
    """One tick per logged timestamp, plus the alarm instant if needed."""
    times = {t for t, _ in plan.intruder}
    for pts in plan.markers.values():
        times.update(t for t, _ in pts)
    if plan.dispatch_at is not None:
        times.add(plan.dispatch_at)
    fired = False
    ticks: list[ReplayTick] = []
    for t in sorted(times):
        person = next((p for ts, p in reversed(plan.intruder) if ts <= t), None)
        markers: dict[str, XY] = {}
        for oid, pts in plan.markers.items():
            pos = next((p for ts, p in reversed(pts) if ts <= t), None)
            if pos is not None:
                markers[oid] = pos
        dispatch = (
            (not fired)
            and plan.dispatch_at is not None
            and t + 1e-9 >= plan.dispatch_at
            and plan.dispatch_xy is not None
        )
        if dispatch:
            fired = True
        ticks.append(ReplayTick(t=t, person=person, markers=markers, dispatch=dispatch))
    return ticks


def densify_intruder(plan: ReplayPlan, dt: float = 0.5) -> ReplayPlan:
    """Resample the logged path plus tactic waypoints so live replay is not two poses."""
    pts: list[XY] = [p for _, p in plan.intruder]
    for waypoint in plan.waypoints:
        if not pts or abs(pts[-1].x - waypoint.x) > 1e-6 or abs(pts[-1].y - waypoint.y) > 1e-6:
            pts.append(waypoint)
    if len(pts) < 2:
        return plan
    speed = plan.speed_mps if plan.speed_mps > 0 else 1.6
    samples = interpolate_path(pts, speed_mps=speed, dt=dt)
    return replace(plan, intruder=[(round(t, 3), xy) for t, xy in samples])


def dispatch_via_mcp(x: float, y: float) -> str:
    """Ask the running WalkModule to auction and walk the Go2."""
    from dimos.agents.mcp.mcp_adapter import McpAdapter

    return McpAdapter.from_run_entry(timeout=60).call_tool_text("dispatch_verify", {"x": x, "y": y})


def run_replay(
    plan: ReplayPlan,
    *,
    live: bool = False,
    realtime_scale: float = 0.0,
    sleep: Callable[[float], None] | None = None,
    dispatch: Callable[[float, float], str] | None = None,
    on_tick: Callable[[ReplayTick], None] | None = None,
    densify: bool | None = None,
) -> dict[str, Any]:
    """Play the plan. `realtime_scale=0` is instant (tests). Live publishes /person_pose."""
    import time
    from math import atan2

    if densify is None:
        densify = live
    if densify:
        plan = densify_intruder(plan)
    sleeper = sleep or time.sleep
    last_t = 0.0
    n_person = 0
    n_dispatch = 0
    result: str | None = None
    prev: XY | None = None
    for tick in iter_ticks(plan):
        if realtime_scale > 0:
            sleeper(max(0.0, (tick.t - last_t) * realtime_scale))
        last_t = tick.t
        if tick.person is not None:
            n_person += 1
            if live:
                yaw = 0.0
                if prev is not None:
                    yaw = atan2(tick.person.y - prev.y, tick.person.x - prev.x)
                from airtight.dimos_lane.person import publish_person_pose

                publish_person_pose(tick.person.x, tick.person.y, yaw_rad=yaw)
                prev = tick.person
        if on_tick is not None:
            on_tick(tick)
        if tick.dispatch and plan.dispatch_xy is not None:
            n_dispatch += 1
            xy = plan.dispatch_xy
            if dispatch is not None:
                result = dispatch(xy.x, xy.y)
            else:
                orch = Orchestrator()
                result = orch.dispatch_verify(xy.x, xy.y)
    return {
        "title": plan.title,
        "person": n_person,
        "dispatch": n_dispatch,
        "dispatch_result": result,
        "timely": plan.timely_detected,
    }


def write_rrd(plan: ReplayPlan, dest: Path) -> Path:
    """Log the intruder path and fleet markers into a Rerun recording."""
    import rerun as rr

    plan = densify_intruder(plan)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rr.init("airtight-replay", recording_id=dest.stem, spawn=False)
    rr.save(str(dest))
    from airtight.dimos_lane.site_io import load_example_site

    site = load_example_site()
    rr.log(
        "world/asset",
        rr.Points3D([[site.asset.x, site.asset.y, 0.5]], colors=[[200, 40, 40]], radii=[1.2]),
        static=True,
    )
    for tick in iter_ticks(plan):
        try:
            rr.set_time("sim", timestamp=tick.t)
        except TypeError:
            rr.set_time("sim", duration=tick.t)
        if tick.person is not None:
            alarmed = tick.dispatch or (plan.t_alarm is not None and tick.t >= plan.t_alarm)
            color = [255, 80, 80] if alarmed else [255, 200, 40]
            rr.log(
                "world/intruder",
                rr.Points3D([[tick.person.x, tick.person.y, 0.4]], colors=[color], radii=[0.7]),
            )
        for oid, xy in tick.markers.items():
            z = 0.3 if "go2" in oid or "guard" in oid else 8.0
            hue = [80, 200, 255] if "go2" in oid else [140, 160, 255]
            rr.log(
                f"world/fleet/{oid}",
                rr.Points3D([[xy.x, xy.y, z]], colors=[hue], radii=[0.6]),
            )
    disconnect = getattr(rr, "disconnect", None)
    if callable(disconnect):
        disconnect()
    return dest


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
