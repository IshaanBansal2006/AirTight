from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path

import pytest

from airtight.contracts import XY, FixedSensor, FleetConfig, SensorCurves, Site
from airtight.score.hardware import (
    CAMERA_COST_KEY,
    DEFAULT_COSTS,
    HARDWARE_DRONES,
    SWAP_CHARGE_TIME_S,
    SWAP_COST_KEY,
    HardwareSpec,
    all_specs,
    build_fleet,
    build_site,
    cost_per_hour,
    load_costs,
    uncovered_entries,
    write_variants,
)

REAL_DIR = Path(__file__).resolve().parents[2] / "data/lane_c_export/scenarios/logistics_yard"


def _example(name: str) -> str:
    return resources.files("airtight.contracts.examples").joinpath(name).read_text()


def _fleet() -> FleetConfig:
    return FleetConfig.model_validate_json(_example("fleet_config.json"))


def _curves() -> SensorCurves:
    return SensorCurves.model_validate_json(_example("sensor_curve.json"))


def _site() -> Site:
    """The example yard with its one camera turned to look at the north gate it sits under."""
    site = Site.model_validate_json(_example("site.json"))
    cam = site.fixed_sensors[0].model_copy(update={"heading_deg": 90.0})
    return site.model_copy(update={"fixed_sensors": [cam]})


def test_twenty_four_specs_with_unique_safe_names_in_a_fixed_order() -> None:
    specs = all_specs()
    assert len(specs) == len(HARDWARE_DRONES) * 2 * 2 == 24
    assert specs == all_specs()
    names = [s.name for s in specs]
    assert len(set(names)) == 24
    assert all(n.replace("_", "").isalnum() for n in names)
    assert specs[0] == HardwareSpec(1, False, False)
    assert specs[-1] == HardwareSpec(6, True, True)
    assert HardwareSpec(3, True, True).name == "d3_swap_cams"
    assert HardwareSpec(2, False, False).name == "d2_std_nocams"


def test_two_standard_drones_reproduce_the_baselines_agents() -> None:
    baseline = _fleet()
    fleet = build_fleet(baseline, HardwareSpec(2, False, False))
    assert fleet.agents == baseline.agents
    assert fleet.name == "d2_std_nocams"
    assert fleet.cost_per_hour() == baseline.cost_per_hour()
    assert fleet.cost_per_hour_by_type == baseline.cost_per_hour_by_type
    assert fleet.comms_mode == baseline.comms_mode
    assert fleet.charge_policy.threshold_frac == baseline.charge_policy.threshold_frac


def test_drones_are_clones_of_the_first_and_the_rest_is_kept_in_order() -> None:
    baseline = _fleet()
    first = next(a for a in baseline.agents if a.type == "drone")
    others = [a for a in baseline.agents if a.type != "drone"]
    for spec in all_specs():
        fleet = build_fleet(baseline, spec)
        FleetConfig.model_validate(fleet.model_dump())
        drones = fleet.agents[: spec.n_drones]
        assert [d.id for d in drones] == [f"drone_{i}" for i in range(1, spec.n_drones + 1)]
        assert fleet.agents[spec.n_drones :] == others
        charge = SWAP_CHARGE_TIME_S if spec.swap_docks else first.charge_time_s
        for d in drones:
            assert d == first.model_copy(update={"id": d.id, "charge_time_s": charge})
        drone_price = baseline.cost_per_hour_by_type["drone"]
        assert fleet.cost_per_hour() == pytest.approx(
            baseline.cost_per_hour() + (spec.n_drones - 2) * drone_price
        )


def test_offsets_are_cleared_and_the_baseline_is_not_mutated() -> None:
    baseline = _fleet()
    policy = baseline.charge_policy.model_copy(update={"stagger_offsets_s": {"drone_2": 900.0}})
    staggered = baseline.model_copy(update={"charge_policy": policy})
    before = staggered.model_dump_json()
    fleet = build_fleet(staggered, HardwareSpec(1, True, False))
    assert fleet.charge_policy.stagger_offsets_s == {}
    assert staggered.model_dump_json() == before


def test_a_baseline_without_a_drone_is_refused() -> None:
    baseline = _fleet()
    ground = baseline.model_copy(
        update={"agents": [a for a in baseline.agents if a.type != "drone"]}
    )
    with pytest.raises(ValueError, match="no drone"):
        build_fleet(ground, HardwareSpec(2, False, False))


def test_coverage_needs_both_range_and_the_wedge() -> None:
    site, curves = _site(), _curves()
    assert uncovered_entries(site, curves) == ["east_fence", "loading_dock"]
    turned = site.model_copy(
        update={"fixed_sensors": [site.fixed_sensors[0].model_copy(update={"heading_deg": 270.0})]}
    )
    assert uncovered_entries(turned, curves) == ["north_gate", "east_fence", "loading_dock"]
    assert uncovered_entries(turned, curves, require_fov=False) == ["east_fence", "loading_dock"]
    edge = site.fixed_sensors[0].model_copy(update={"heading_deg": 90.0 + 30.0})
    assert "north_gate" not in uncovered_entries(
        site.model_copy(update={"fixed_sensors": [edge]}), curves
    )
    far = site.fixed_sensors[0].model_copy(update={"position": XY(x=60.0, y=44.0)})
    assert "north_gate" in uncovered_entries(
        site.model_copy(update={"fixed_sensors": [far]}), curves
    )


def test_a_sensor_on_the_entry_point_covers_it_whatever_its_heading() -> None:
    site, curves = _site(), _curves()
    on_it = FixedSensor(
        id="cam_east",
        position=site.entry("east_fence").position,
        sensor_type="fixed_camera",
        heading_deg=0.0,
    )
    both = site.model_copy(update={"fixed_sensors": [*site.fixed_sensors, on_it]})
    assert uncovered_entries(both, curves) == ["loading_dock"]


def test_dead_outer_bins_shrink_the_footprint() -> None:
    site, curves = _site(), _curves()
    cam = curves.curves["fixed_camera"]
    near = site.fixed_sensors[0].model_copy(update={"position": XY(x=60.0, y=66.0)})
    moved = site.model_copy(update={"fixed_sensors": [near]})
    assert "north_gate" not in uncovered_entries(moved, curves)
    dead = cam.model_copy(update={"pd_per_look": [cam.pd_per_look[0], 0.0, 0.0, 0.0]})
    short = curves.model_copy(update={"curves": {**curves.curves, "fixed_camera": dead}})
    assert "north_gate" in uncovered_entries(moved, short)


def test_cameras_are_the_only_change_to_the_site() -> None:
    site, curves = _site(), _curves()
    variant = build_site(site, curves, HardwareSpec(2, False, True))
    assert variant.fixed_sensors[: len(site.fixed_sensors)] == site.fixed_sensors
    added = variant.fixed_sensors[len(site.fixed_sensors) :]
    assert [s.id for s in added] == ["cam_entry_east_fence", "cam_entry_loading_dock"]
    for sensor in added:
        entry = site.entry(sensor.id.removeprefix("cam_entry_"))
        assert sensor.position == entry.position
        assert sensor.sensor_type == "fixed_camera"
        assert 0.0 <= sensor.heading_deg < 360.0
        to_asset = math.atan2(site.asset.y - entry.position.y, site.asset.x - entry.position.x)
        assert math.cos(math.radians(sensor.heading_deg) - to_asset) == pytest.approx(1.0)
    assert added[0].heading_deg == pytest.approx(180.0)
    restored = variant.model_copy(update={"fixed_sensors": site.fixed_sensors})
    assert restored.model_dump_json() == site.model_dump_json()
    for field in (
        "name",
        "bounds",
        "perimeter",
        "entry_points",
        "asset",
        "response_time_s",
        "docks",
        "benign_routes",
    ):
        assert getattr(variant, field) == getattr(site, field)
    assert uncovered_entries(variant, curves) == []
    assert build_site(variant, curves, HardwareSpec(2, False, True)) == variant


def test_without_cameras_the_site_is_unchanged() -> None:
    site, curves = _site(), _curves()
    same = build_site(site, curves, HardwareSpec(4, True, False))
    assert same.content_hash() == site.content_hash()
    assert same.model_dump_json() == site.model_dump_json()


def test_no_camera_type_to_copy_or_no_curve_is_refused() -> None:
    site, curves = _site(), _curves()
    spec = HardwareSpec(2, False, True)
    with pytest.raises(ValueError, match="no fixed sensor"):
        build_site(site.model_copy(update={"fixed_sensors": []}), curves, spec)
    no_curve = curves.model_copy(
        update={"curves": {k: v for k, v in curves.curves.items() if k != "fixed_camera"}}
    )
    with pytest.raises(ValueError, match="no sensor curve"):
        build_site(site, no_curve, spec)


def test_default_costs_show_their_arithmetic_and_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "campaign" / "costs.json"
    costs = load_costs(path)
    assert path.exists()
    assert json.loads(path.read_text()) == DEFAULT_COSTS
    assert set(costs) == {SWAP_COST_KEY, CAMERA_COST_KEY}
    for name, entry in DEFAULT_COSTS.items():
        assert costs[name] == entry["value"] >= 0
        assert entry["units"] == "USD per hour"
        assert entry["status"] == "verify"
        assert "26,280" in entry["source"]
        assert f"{entry['value']:.2f} $/h" in entry["source"]
    assert 1.0 <= costs[SWAP_COST_KEY] <= 2.0
    assert 0.3 <= costs[CAMERA_COST_KEY] <= 0.6


def test_an_existing_costs_file_wins_and_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "costs.json"
    edited = json.loads(json.dumps(DEFAULT_COSTS))
    edited[CAMERA_COST_KEY]["value"] = 0.5
    edited[CAMERA_COST_KEY]["status"] = "verified"
    path.write_text(json.dumps(edited))
    assert load_costs(path)[CAMERA_COST_KEY] == 0.5
    assert json.loads(path.read_text()) == edited
    for field, bad in (("value", -1.0), ("value", "free"), ("source", " "), ("status", "guess")):
        broken = json.loads(json.dumps(DEFAULT_COSTS))
        broken[SWAP_COST_KEY][field] = bad
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=SWAP_COST_KEY):
            load_costs(path)
    path.write_text(json.dumps({SWAP_COST_KEY: DEFAULT_COSTS[SWAP_COST_KEY]}))
    with pytest.raises(ValueError, match="missing"):
        load_costs(path)


def test_cost_adds_the_swap_price_per_drone_and_the_price_of_each_added_camera() -> None:
    baseline, site, curves = _fleet(), _site(), _curves()
    costs = {SWAP_COST_KEY: 1.5, CAMERA_COST_KEY: 0.25}
    drone_price = baseline.cost_per_hour_by_type["drone"]
    for spec in all_specs():
        fleet = build_fleet(baseline, spec)
        variant = build_site(site, curves, spec)
        expected = baseline.cost_per_hour() + (spec.n_drones - 2) * drone_price
        if spec.swap_docks:
            expected += 1.5 * spec.n_drones
        if spec.entry_cameras:
            expected += 0.25 * 2
        assert cost_per_hour(fleet, variant, site, spec, costs) == pytest.approx(expected)
    plain = HardwareSpec(2, False, False)
    assert cost_per_hour(build_fleet(baseline, plain), site, site, plain, costs) == (
        baseline.cost_per_hour()
    )


def test_written_variants_round_trip_through_the_contracts(tmp_path: Path) -> None:
    baseline, site, curves = _fleet(), _site(), _curves()
    specs = all_specs()
    paths = write_variants(tmp_path / "hw", baseline, site, curves, specs)
    assert list(paths) == [s.name for s in specs]
    for spec in specs:
        fleet_path, site_path = paths[spec.name]
        assert fleet_path == tmp_path / "hw" / "fleets" / f"{spec.name}.json"
        assert site_path == tmp_path / "hw" / "sites" / f"{spec.name}.json"
        assert FleetConfig.model_validate_json(fleet_path.read_text()) == build_fleet(
            baseline, spec
        )
        assert Site.model_validate_json(site_path.read_text()) == build_site(site, curves, spec)


def test_the_real_logistics_yard() -> None:
    if not REAL_DIR.is_dir():
        pytest.skip("data/lane_c_export is not checked in")
    site = Site.model_validate_json((REAL_DIR / "site.json").read_text())
    curves = SensorCurves.model_validate_json((REAL_DIR / "sensor_curve.json").read_text())
    sweep = json.loads((REAL_DIR / "fleets" / "sweep.json").read_text())
    baseline = FleetConfig.model_validate_json(
        (REAL_DIR / "fleets" / f"{sweep['baseline']}.json").read_text()
    )
    assert build_fleet(baseline, HardwareSpec(2, False, False)).agents == baseline.agents
    outer = ["staff_gate", "service_gate", "rear_fence_gap"]
    assert uncovered_entries(site, curves, require_fov=False) == outer
    # cam_main_gate is 3 m inside the gate and looks into the yard (270 degrees), so the gate
    # point itself is behind it: in range, outside the wedge.
    assert uncovered_entries(site, curves) == ["main_gate", *outer]
    variant = build_site(site, curves, HardwareSpec(2, False, True))
    assert len(variant.fixed_sensors) == len(site.fixed_sensors) + 4
    assert (
        len(build_site(site, curves, HardwareSpec(2, False, True), require_fov=False).fixed_sensors)
        == len(site.fixed_sensors) + 3
    )
    restored = variant.model_copy(update={"fixed_sensors": site.fixed_sensors})
    assert restored.content_hash() == site.content_hash()
