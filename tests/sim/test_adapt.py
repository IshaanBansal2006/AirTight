from __future__ import annotations

import math
from importlib import resources

import numpy as np
import pytest

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
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


def test_assets_is_the_single_contract_asset(site: Site) -> None:
    assert adapt.assets(site).tolist() == [[site.asset.x, site.asset.y]]


def test_start_positions_distinct_for_four_agents_sharing_two_docks(
    site: Site, fleet: FleetConfig
) -> None:
    assert len(fleet.agents) == 4 and len(site.docks) == 2
    starts = [adapt.start_position(site, fleet, a) for a in fleet.agents]
    for i in range(4):
        dock = site.docks[i % 2].position
        assert math.isclose(
            float(np.linalg.norm(starts[i] - np.array([dock.x, dock.y]))), adapt.START_OFFSET_M
        )
        for j in range(i + 1, 4):
            assert float(np.linalg.norm(starts[i] - starts[j])) > 1.0


def test_start_position_falls_back_to_perimeter_centroid(site: Site, fleet: FleetConfig) -> None:
    no_docks = site.model_copy(update={"docks": []})
    start = adapt.start_position(no_docks, fleet, fleet.agents[0])
    assert start.tolist() == [60.0 + adapt.START_OFFSET_M, 40.0]


def test_start_position_rejects_agent_outside_fleet(site: Site, fleet: FleetConfig) -> None:
    stranger = fleet.agents[0].model_copy(update={"id": "stranger"})
    with pytest.raises(ValueError, match="stranger"):
        adapt.start_position(site, fleet, stranger)


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
    for agent in fleet.agents:
        assert np.array_equal(
            adapt.start_position(site, fleet, agent), adapt.start_position(site, fleet, agent)
        )
    assert adapt.t_cdp(site, tactic) == adapt.t_cdp(site, tactic)
