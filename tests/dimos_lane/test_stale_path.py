from __future__ import annotations

from pathlib import Path

from airtight.contracts import XY
from airtight.dimos_lane.clips import charging_window_tactic
from airtight.dimos_lane.site_io import (
    load_logistics_curves,
    load_logistics_fleet,
    load_logistics_site,
)
from airtight.dimos_lane.stale_path import (
    CellNeed,
    charging_phase,
    charging_window_from_staleness,
    waypoints_through_stale,
)


def test_waypoints_use_the_stalest_cell_in_each_band() -> None:
    site = load_logistics_site()
    entry = site.entry("rear_fence_gap").position
    asset = site.asset
    early = XY(x=entry.x + 0.3 * (asset.x - entry.x), y=entry.y + 0.3 * (asset.y - entry.y))
    late = XY(x=entry.x + 0.7 * (asset.x - entry.x), y=entry.y + 0.7 * (asset.y - entry.y))
    decoy_late = XY(x=late.x + 8.0, y=late.y)
    cells = [(early, 10.0), (late, 400.0), (decoy_late, 50.0)]
    path = waypoints_through_stale(site, entry, cells)
    assert path[-1] == asset
    assert late in path
    assert decoy_late not in path


def test_high_value_beats_an_empty_stale_corner() -> None:
    site = load_logistics_site()
    entry = site.entry("rear_fence_gap").position
    asset = site.asset
    near = XY(x=entry.x + 0.7 * (asset.x - entry.x), y=entry.y + 0.7 * (asset.y - entry.y))
    empty = XY(x=150.0, y=10.0)
    cells = [
        CellNeed(empty, stale_s=5000.0, value=0.3),
        CellNeed(near, stale_s=80.0, value=1.2),
    ]
    path = waypoints_through_stale(site, entry, cells)
    assert near in path
    assert empty not in path


def test_charging_phase_is_when_the_drones_are_down() -> None:
    fleet = load_logistics_fleet("d2_go2_guard_sync")
    phase = charging_phase(fleet)
    assert 0.0 <= phase < 1.0


def test_charging_window_waypoints_come_from_staleness() -> None:
    site = load_logistics_site()
    fleet = load_logistics_fleet("d2_go2_guard_sync")
    curves = load_logistics_curves()
    tactic = charging_window_from_staleness(site, fleet, curves)
    start = site.entry(tactic.entry_id).position
    assert tactic.family == "charging_window"
    assert tactic.waypoints[-1] == site.asset
    assert len(tactic.waypoints) >= 2
    elbow = XY(x=start.x, y=site.asset.y)
    assert tactic.waypoints[0] != elbow
    assert all(p.y > 20.0 for p in tactic.waypoints[:-1])


def test_charging_window_tactic_autogens_without_c_file() -> None:
    site = load_logistics_site()
    tactic = charging_window_tactic(site, tactics_dir=Path("/no/tactics"))
    assert tactic.id == "charging_window-a9-stale"
    assert tactic.waypoints[-1] == site.asset
