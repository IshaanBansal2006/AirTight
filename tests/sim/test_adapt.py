from __future__ import annotations

import math
from importlib import resources

import numpy as np
import pytest

from airtight.contracts import XY, Decoy, FleetConfig, SensorCurves, Site, Tactic
from airtight.sim import adapt
from airtight.sim.runner import _path_length_m

EXAMPLES = resources.files("airtight.contracts.examples")
TACTIC_FILES = sorted(
    f.name for f in EXAMPLES.iterdir() if f.name.startswith("tactic") and f.name.endswith(".json")
)


def _load(name: str, model: type):  # type: ignore[no-untyped-def]
    return model.model_validate_json(EXAMPLES.joinpath(name).read_text())


@pytest.fixture
def site() -> Site:
    return _load("site.json", Site)


@pytest.fixture
def fleet() -> FleetConfig:
    return _load("fleet_config.json", FleetConfig)


def test_there_is_at_least_one_example_tactic() -> None:
    assert "tactic.json" in TACTIC_FILES


@pytest.mark.parametrize("filename", TACTIC_FILES)
def test_intruder_path_matches_stub(site: Site, filename: str) -> None:
    tactic = _load(filename, Tactic)
    path = adapt.intruder_path(site, tactic)
    entry = site.entry(tactic.entry_id).position
    assert path.shape == (len(tactic.waypoints) + 1, 2)
    assert tuple(path[0]) == (entry.x, entry.y)
    assert tuple(path[-1]) == (tactic.waypoints[-1].x, tactic.waypoints[-1].y)
    assert adapt.path_length_m(path) == _path_length_m(site, tactic)
    assert adapt.t_reach(site, tactic) == _path_length_m(site, tactic) / tactic.speed_mps


def test_unknown_entry_id_raises(site: Site) -> None:
    tactic = _load("tactic.json", Tactic).model_copy(update={"entry_id": "nope"})
    with pytest.raises(KeyError, match="nope"):
        adapt.intruder_path(site, tactic)


def test_t_cdp_on_contract_example_is_zero(site: Site) -> None:
    # Known degenerate example: the path takes about 22 s and response_time_s is 90, so the
    # critical detection point clamps to 0 and no alarm can be timely. Raised with the team;
    # lane B does not work around it by touching contracts.
    tactic = _load("tactic.json", Tactic)
    assert adapt.t_reach(site, tactic) < site.response_time_s
    assert adapt.t_cdp(site, tactic) == 0.0


def test_t_cdp_is_t_reach_minus_response_time_when_positive(site: Site) -> None:
    tactic = _load("tactic.json", Tactic)
    quick = site.model_copy(update={"response_time_s": 5.0})
    assert adapt.t_cdp(quick, tactic) == adapt.t_reach(quick, tactic) - 5.0


def test_bounds_are_plain_floats(site: Site) -> None:
    assert adapt.bounds(site) == (0.0, 0.0, 120.0, 80.0)
    assert all(type(v) is float for v in adapt.bounds(site))


def test_perimeter_is_an_ordered_float_array(site: Site) -> None:
    perimeter = adapt.perimeter(site)
    assert perimeter.dtype == np.float64
    assert perimeter.tolist() == [[5.0, 5.0], [115.0, 5.0], [115.0, 75.0], [5.0, 75.0]]


def test_assets_is_the_single_contract_asset(site: Site) -> None:
    assert adapt.assets(site).tolist() == [[site.asset.x, site.asset.y]]


def test_start_positions_distinct_for_four_agents_sharing_two_docks(
    site: Site, fleet: FleetConfig
) -> None:
    assert len(fleet.agents) == 4 and len(site.docks) == 2
    starts = [adapt.start_position(site, fleet, a) for a in adapt.agent_ids(fleet)]
    for i in range(4):
        dock = site.docks[i % 2].position
        assert math.isclose(
            float(np.linalg.norm(starts[i] - np.array([dock.x, dock.y]))), adapt.START_OFFSET_M
        )
        for j in range(i + 1, 4):
            assert float(np.linalg.norm(starts[i] - starts[j])) > 1.0


def test_start_position_falls_back_to_perimeter_centroid(site: Site, fleet: FleetConfig) -> None:
    no_docks = site.model_copy(update={"docks": []})
    start = adapt.start_position(no_docks, fleet, "drone_1")
    assert start.tolist() == [60.0 + adapt.START_OFFSET_M, 40.0]


def test_start_position_rejects_agent_outside_fleet(site: Site, fleet: FleetConfig) -> None:
    with pytest.raises(ValueError, match="stranger"):
        adapt.start_position(site, fleet, "stranger")


def test_benign_speed_known_and_default() -> None:
    assert adapt.benign_speed("person") == 1.4
    assert adapt.benign_speed("vehicle") == 5.0
    assert adapt.benign_speed("unheard_of") == adapt.DEFAULT_BENIGN_SPEED_MPS


def test_every_example_benign_class_has_an_explicit_speed(site: Site) -> None:
    assert {r.cls for r in site.benign_routes} <= set(adapt.BENIGN_SPEED_MPS)


def test_sensor_range_and_look_rate() -> None:
    curves = _load("sensor_curve.json", SensorCurves)
    assert adapt.sensor_max_range_m(curves, "drone_camera") == 40.0
    assert adapt.sensor_look_rate_hz(curves, "human_eye") == 0.5
    with pytest.raises(KeyError, match="sonar"):
        adapt.sensor_max_range_m(curves, "sonar")


def test_everything_is_deterministic(site: Site, fleet: FleetConfig) -> None:
    tactic = _load("tactic.json", Tactic)
    assert np.array_equal(adapt.intruder_path(site, tactic), adapt.intruder_path(site, tactic))
    for agent_id in adapt.agent_ids(fleet):
        assert np.array_equal(
            adapt.start_position(site, fleet, agent_id),
            adapt.start_position(site, fleet, agent_id),
        )
    assert adapt.t_cdp(site, tactic) == adapt.t_cdp(site, tactic)


def test_agent_accessors_follow_fleet_order(fleet: FleetConfig) -> None:
    assert adapt.agent_ids(fleet) == ["drone_1", "drone_2", "go2_1", "guard_1"]
    assert adapt.agent_speed_mps(fleet, "drone_1") == 8.0
    assert adapt.agent_speed_mps(fleet, "go2_1") == 1.2
    assert adapt.agent_sensor_type(fleet, "guard_1") == "human_eye"
    with pytest.raises(ValueError, match="stranger"):
        adapt.agent_speed_mps(fleet, "stranger")


def test_footprint_radius_is_the_last_bin_with_positive_pd() -> None:
    curves = _load("sensor_curve.json", SensorCurves)
    assert adapt.sensor_footprint_radius_m(curves, "drone_camera") == 40.0
    drone = curves.curves["drone_camera"]
    fading = drone.model_copy(update={"pd_per_look": [0.95, 0.9, 0.8, 0.6, 0.0, 0.0]})
    blind = drone.model_copy(update={"pd_per_look": [0.0] * 6})
    patched = curves.model_copy(update={"curves": {"drone_camera": fading, "blind": blind}})
    assert adapt.sensor_footprint_radius_m(patched, "drone_camera") == 20.0
    assert adapt.sensor_max_range_m(patched, "drone_camera") == 40.0  # distinct on purpose
    with pytest.raises(ValueError, match="blind"):
        adapt.sensor_footprint_radius_m(patched, "blind")


def test_sensor_fov_comes_from_the_contract() -> None:
    curves = _load("sensor_curve.json", SensorCurves)
    assert adapt.sensor_fov_deg(curves, "drone_camera") == 70.0
    assert adapt.sensor_fov_deg(curves, "human_eye") == 120.0


def test_benign_routes_are_plain_records_with_class_speeds(site: Site) -> None:
    routes = adapt.benign_routes(site)
    assert [r.route_id for r in routes] == ["delivery", "staff_walk"]
    delivery, walk = routes
    assert (delivery.cls, delivery.arrivals_per_hour, delivery.speed_mps) == ("vehicle", 4.0, 5.0)
    assert (walk.cls, walk.arrivals_per_hour, walk.speed_mps) == ("person", 10.0, 1.4)
    assert delivery.points.dtype == np.float64
    assert delivery.points.tolist() == [[20.0, 5.0], [20.0, 30.0], [20.0, 5.0]]
    assert adapt.benign_routes(site.model_copy(update={"benign_routes": []})) == []


def test_decoy_spec_is_on_the_episode_clock() -> None:
    tactic = _load("tactic.json", Tactic)
    assert adapt.decoy_spec(tactic) is None
    lure = tactic.model_copy(update={"decoy": Decoy(position=XY(x=100, y=60), lead_time_s=40)})
    spec = adapt.decoy_spec(lure)
    assert spec is not None
    assert spec.position.tolist() == [100.0, 60.0]
    assert (spec.t_on, spec.t_off) == (-40.0, 20.0)
    assert adapt.intruder_speed_mps(tactic) == 1.6


def test_pd_per_look_by_range_bin() -> None:
    curves = _load("sensor_curve.json", SensorCurves)
    assert adapt.pd_per_look(curves, "drone_camera", 0.0) == 0.95
    assert adapt.pd_per_look(curves, "drone_camera", 5.0) == 0.95  # an edge closes its own bin
    assert adapt.pd_per_look(curves, "drone_camera", 5.01) == 0.9
    assert adapt.pd_per_look(curves, "drone_camera", 40.0) == 0.1
    assert adapt.pd_per_look(curves, "drone_camera", 40.01) == 0.0


def test_true_fp_per_look_is_the_contract_scalar() -> None:
    curves = _load("sensor_curve.json", SensorCurves)
    assert adapt.true_fp_per_look(curves, "drone_camera", "person") == 0.03
    assert adapt.true_fp_per_look(curves, "human_eye", "vehicle") == 0.002
    with pytest.raises(KeyError, match="unicorn"):
        adapt.true_fp_per_look(curves, "drone_camera", "unicorn")


def test_response_time_critical_radius_and_docks(site: Site) -> None:
    assert adapt.response_time_s(site) == 90.0
    assert adapt.critical_radius_m(site, 2.5) == 225.0
    assert adapt.docks(site).tolist() == [[10.0, 70.0], [110.0, 10.0]]
    assert adapt.docks(site.model_copy(update={"docks": []})).shape == (0, 2)


def test_fixed_sensors_convert_heading_to_radians(site: Site) -> None:
    (cam,) = adapt.fixed_sensors(site)
    assert (cam.sensor_id, cam.sensor_type) == ("cam_north", "fixed_camera")
    assert cam.position.tolist() == [60.0, 72.0]
    # compare with the example itself: the team may correct its heading (lane C has)
    assert cam.heading_rad == pytest.approx(math.radians(site.fixed_sensors[0].heading_deg))
    turned = site.fixed_sensors[0].model_copy(update={"heading_deg": 270.0})
    (west,) = adapt.fixed_sensors(site.model_copy(update={"fixed_sensors": [turned]}))
    assert west.heading_rad == pytest.approx(3 * math.pi / 2)


def test_has_curve_and_fp_classes() -> None:
    curves = _load("sensor_curve.json", SensorCurves)
    assert adapt.has_curve(curves, "drone_camera") and not adapt.has_curve(curves, "sonar")
    assert adapt.fp_classes(curves, "drone_camera") == {"person", "vehicle"}


def test_agent_energy_offsets_and_never_charges(fleet: FleetConfig) -> None:
    drone = adapt.agent_energy(fleet, "drone_1")
    assert drone == adapt.EnergySpec(endurance_s=1500.0, charge_time_s=2400.0, offset_s=0.0)
    assert adapt.agent_energy(fleet, "guard_1") is None  # charge_time_s == 0 means never charges
    offsets = fleet.charge_policy.model_copy(update={"stagger_offsets_s": {"drone_2": 1950.0}})
    staggered = fleet.model_copy(update={"charge_policy": offsets})
    spec = adapt.agent_energy(staggered, "drone_2")
    assert spec is not None and spec.offset_s == 1950.0
    assert adapt.agent_energy(staggered, "drone_1") == drone


def test_reference_cycle_is_lane_cs_definition(fleet: FleetConfig) -> None:
    # redteam/families.py charge_cycle_s: mean of endurance + charge over agents that charge
    expected = float(
        np.mean([a.endurance_s + a.charge_time_s for a in fleet.agents if a.charge_time_s > 0])
    )
    assert adapt.reference_cycle_s(fleet) == expected == 5600.0  # (3900 + 3900 + 9000) / 3
    guards_only = fleet.model_copy(update={"agents": [fleet.agents[3]]})
    assert adapt.reference_cycle_s(guards_only) == 3600.0  # and the same fallback


def test_tactic_phase() -> None:
    assert adapt.tactic_phase(_load("tactic.json", Tactic)) == 0.62
