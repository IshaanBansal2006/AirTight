"""Intruder waypoints from neglected high-value cells.

Charging-window clips do not take a searched path from lane C. Phase is the drone
charging gap. Intermediate waypoints are patrol cells that are both valuable to
protect (asset-weighted) and stale (not surveilled recently).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median
from typing import TYPE_CHECKING, NamedTuple

from airtight.contracts import XY, Tactic, TacticFamily
from airtight.sim import adapt
from airtight.sim.coverage import uncovered_intervals
from airtight.sim.episode import _run_loop, official_params

if TYPE_CHECKING:
    from airtight.contracts import FleetConfig, SensorCurves, Site
    from airtight.sim.fleet import PatrolController

MIN_LEG_M = 2.0
N_STALE_WAYPOINTS = 2
ASSET_SCALE_M = 25.0
ASSET_BASE = 0.3
HANDOFF_FAMILY: TacticFamily = "charging_window"
HANDOFF_ENTRY = "rear_fence_gap"
HANDOFF_SPEED_MPS = 1.8301223906650246


class CellNeed(NamedTuple):
    xy: XY
    stale_s: float
    value: float

    @property
    def score(self) -> float:
        """Neglected high-value: unwatched recently, and worth protecting."""
        return self.stale_s * self.value


def charging_phase(fleet: FleetConfig) -> float:
    """Midpoint of the longest interval when every drone is off the yard."""
    drone_ids = [a.id for a in fleet.agents if a.type == "drone"]
    gaps = uncovered_intervals(fleet, only=drone_ids) or uncovered_intervals(fleet)
    if not gaps:
        return 0.99
    gap = max(gaps, key=lambda g: g.duration_s)
    return 0.5 * (gap.start_phase + gap.end_phase)


def _progress(point: XY, start: XY, end: XY) -> float:
    vx, vy = end.x - start.x, end.y - start.y
    length = math.hypot(vx, vy)
    if length == 0.0:
        return 0.0
    return ((point.x - start.x) * vx + (point.y - start.y) * vy) / (length * length)


def _dist(a: XY, b: XY) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def asset_value(site: Site, point: XY) -> float:
    """Same shape as the sim's asset patrol weight: high near the thing being protected."""
    return ASSET_BASE + math.exp(-_dist(point, site.asset) / ASSET_SCALE_M)


def _as_need(site: Site, cell: CellNeed | tuple[XY, float] | tuple[XY, float, float]) -> CellNeed:
    if isinstance(cell, CellNeed):
        return cell
    if len(cell) == 3:
        return CellNeed(cell[0], cell[1], cell[2])
    return CellNeed(cell[0], cell[1], asset_value(site, cell[0]))


def _need_score(cell: CellNeed, stale_max: float, value_max: float) -> float:
    stale = cell.stale_s / stale_max if stale_max else 0.0
    value = cell.value / value_max if value_max else 0.0
    return stale * value


def snapshot_staleness(
    site: Site, fleet: FleetConfig, curves: SensorCurves, phase: float, seed: int = 0
) -> list[CellNeed]:
    """Patrol cells with staleness (s) and asset-protection weight at episode t=0."""
    captured: dict[str, object] = {}

    def probe(t: float, _agents: object, controller: PatrolController) -> None:
        if captured or t > 0.5:
            return
        stale = controller.staleness(t)
        centres = controller.grid.cell_centers()
        weight = controller.weight
        captured["cells"] = [
            CellNeed(
                XY(x=float(centres[iy, ix, 0]), y=float(centres[iy, ix, 1])),
                float(stale[iy, ix]),
                float(weight[iy, ix]),
            )
            for iy in range(stale.shape[0])
            for ix in range(stale.shape[1])
            if weight[iy, ix] > 0 and stale[iy, ix] > 0
        ]

    t0_abs = phase * adapt.reference_cycle_s(fleet)
    _run_loop(site, fleet, curves, seed, official_params(), [], 1.0, t0_abs, probe=probe)
    cells = captured.get("cells")
    return list(cells) if isinstance(cells, list) else []


def waypoints_through_stale(
    site: Site,
    entry: XY,
    cells: Sequence[CellNeed | tuple[XY, float] | tuple[XY, float, float]],
    n: int = N_STALE_WAYPOINTS,
) -> list[XY]:
    """One neglected high-value cell per progress band toward the asset, then the asset."""
    asset = site.asset
    corridor = [
        need
        for cell in cells
        for need in [_as_need(site, cell)]
        if 0.12 < _progress(need.xy, entry, asset) < 0.88 and _dist(need.xy, entry) >= MIN_LEG_M
    ]
    if not corridor:
        return [asset]
    floor = median(need.value for need in corridor)
    high_value = [need for need in corridor if need.value >= floor] or corridor
    picks: list[XY] = []
    for band in range(n):
        lo, hi = band / n, (band + 1) / n
        bucket = [need for need in high_value if lo <= _progress(need.xy, entry, asset) < hi]
        if not bucket:
            continue
        stale_max = max(need.stale_s for need in bucket)
        value_max = max(need.value for need in bucket)
        candidate = max(bucket, key=lambda need: _need_score(need, stale_max, value_max)).xy
        if picks and _dist(candidate, picks[-1]) < MIN_LEG_M:
            continue
        if _dist(candidate, asset) < MIN_LEG_M:
            continue
        picks.append(candidate)
    return [*picks, asset] if picks else [asset]


def charging_window_from_staleness(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    *,
    entry_id: str = HANDOFF_ENTRY,
) -> Tactic:
    """Valid charging_window through high-value cells the fleet has not watched recently."""
    entry = site.entry(entry_id).position
    phase = charging_phase(fleet)
    cells = snapshot_staleness(site, fleet, curves, phase)
    waypoints = waypoints_through_stale(site, entry, cells)
    return Tactic(
        id="charging_window-a9-stale",
        family=HANDOFF_FAMILY,
        entry_id=entry_id,
        phase=phase,
        speed_mps=HANDOFF_SPEED_MPS,
        waypoints=waypoints,
        origin="hand",
    )
