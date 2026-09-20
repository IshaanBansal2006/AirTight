from __future__ import annotations

import json
from pathlib import Path

import pytest

from airtight.contracts import XY, AgentSpec, EpisodeResult, FleetConfig, Tactic
from airtight.dimos_lane.clips import charging_window_tactic, find_miss_catch_pair
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
from airtight.redteam.validate import validate


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


def test_empty_or_off_corridor_cells_fall_back_to_the_asset() -> None:
    site = load_logistics_site()
    entry = site.entry("rear_fence_gap").position
    assert waypoints_through_stale(site, entry, []) == [site.asset]
    assert waypoints_through_stale(site, entry, [(entry, 999.0)]) == [site.asset]
    assert waypoints_through_stale(site, entry, [(site.asset, 999.0)]) == [site.asset]


def test_skips_a_second_waypoint_closer_than_min_leg() -> None:
    site = load_logistics_site()
    entry = site.entry("rear_fence_gap").position
    asset = site.asset
    def along(frac: float) -> XY:
        return XY(
            x=entry.x + frac * (asset.x - entry.x),
            y=entry.y + frac * (asset.y - entry.y),
        )
    first = along(0.49)
    too_close = along(0.51)
    path = waypoints_through_stale(
        site,
        entry,
        [
            CellNeed(first, stale_s=100.0, value=1.0),
            CellNeed(too_close, stale_s=100.0, value=1.0),
        ],
    )
    assert path == [first, asset]


def test_charging_phase_when_nobody_docks() -> None:
    fleet = FleetConfig(
        name="guard_only",
        agents=[
            AgentSpec(
                id="guard_1",
                type="guard",
                speed_mps=1.4,
                endurance_s=28800.0,
                charge_time_s=0.0,
                sensor_type="human_eye",
            )
        ],
        cost_per_hour_by_type={"guard": 32.0},
    )
    assert charging_phase(fleet) == 0.99


def test_autogen_tactic_passes_the_redteam_validator() -> None:
    site = load_logistics_site()
    fleet = load_logistics_fleet("d2_go2_guard_sync")
    curves = load_logistics_curves()
    tactic = charging_window_from_staleness(site, fleet, curves)
    assert 0.0 <= tactic.phase < 1.0
    assert validate(tactic, site) == []


def test_prefers_lane_c_searched_tactic_when_present(tmp_path: Path) -> None:
    site = load_logistics_site()
    searched = Tactic(
        id="charging_window-from-c",
        family="charging_window",
        entry_id="rear_fence_gap",
        phase=0.42,
        speed_mps=1.5,
        waypoints=[site.asset],
        origin="search",
    )
    (tmp_path / "top_charging_window.json").write_text(
        json.dumps({"tactics": [json.loads(searched.model_dump_json())]})
    )
    got = charging_window_tactic(site, tactics_dir=tmp_path)
    assert got.id == "charging_window-from-c"
    assert got.origin == "search"


def test_find_miss_catch_pair_raises_when_every_seed_catches() -> None:
    site = load_logistics_site()
    fleet = load_logistics_fleet("d2_go2_guard_sync")
    curves = load_logistics_curves()
    tactic = Tactic(
        id="walk",
        family="charging_window",
        entry_id="rear_fence_gap",
        phase=0.1,
        speed_mps=1.5,
        waypoints=[site.asset],
    )

    def always_catch(*_args: object, **_kwargs: object) -> EpisodeResult:
        return EpisodeResult(timely_detected=True, t_alarm=1.0, t_cdp=10.0, log_path=Path("x.jsonl"))

    with pytest.raises(RuntimeError, match="no seed"):
        find_miss_catch_pair(
            site, fleet, fleet, tactic, curves, [1, 2], Path("/tmp/clip_logs"), always_catch
        )
