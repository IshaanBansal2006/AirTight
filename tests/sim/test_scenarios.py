from __future__ import annotations

import json

import pytest

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
from airtight.sim import adapt, scenarios
from airtight.sim.episode import EpisodeParams, check_setup

MODEL_BY_PREFIX = {
    "site": Site,
    "sensor_curve": SensorCurves,
    "fleet": FleetConfig,
    "tactic": Tactic,
}


def _files(scenario: str) -> list[str]:
    return sorted(p.name for p in scenarios.scenario_dir(scenario).iterdir())


@pytest.mark.parametrize("scenario", scenarios.scenario_names())
def test_every_file_in_every_scenario_validates_against_its_contract_model(scenario: str) -> None:
    files = _files(scenario)
    assert "site.json" in files and "sensor_curve.json" in files
    for name in files:
        prefix = next(
            (p for p in MODEL_BY_PREFIX if name == f"{p}.json" or name.startswith(f"{p}_")), None
        )
        assert prefix is not None, f"{scenario}/{name} is not a recognised scenario file"
        text = scenarios.scenario_dir(scenario).joinpath(name).read_text()
        model = MODEL_BY_PREFIX[prefix].model_validate_json(text)
        assert json.loads(model.model_dump_json()) == json.loads(
            text
        )  # nothing dropped or defaulted


@pytest.mark.parametrize("scenario", scenarios.scenario_names())
def test_every_fleet_and_tactic_is_runnable_on_its_site(scenario: str) -> None:
    site, curves = scenarios.load_site(scenario), scenarios.load_sensor_curves(scenario)
    assert scenarios.names("fleet", scenario) and scenarios.names("tactic", scenario)
    for fleet_name in scenarios.names("fleet", scenario):
        check_setup(site, scenarios.load_fleet(fleet_name, scenario), curves, EpisodeParams())
    for tactic_name in scenarios.names("tactic", scenario):
        tactic = scenarios.load_tactic(tactic_name, scenario)
        assert adapt.intruder_path(site, tactic)[-1].tolist() == adapt.assets(site)[0].tolist()


def test_yard_night_is_what_the_team_agreed() -> None:
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    assert scenarios.scenario_names() == ["yard_night"]
    assert adapt.bounds(site) == (0.0, 0.0, 300.0, 200.0) and adapt.response_time_s(site) == 25.0
    assert scenarios.names("fleet") == [
        "1drone",
        "2drones",
        "2drones_staggered",
        "3drones_staggered",
        "4drones",
    ]
    assert [len(scenarios.load_fleet(n).agents) for n in scenarios.names("fleet")] == [
        1,
        2,
        2,
        3,
        4,
    ]
    thirds = scenarios.load_fleet("3drones_staggered")
    assert thirds.charge_policy.stagger_offsets_s == {
        "d1": 1200.0,
        "d2": 2400.0,
    }  # thirds of 3600 s
    assert {
        a.model_copy(update={"id": "x"})
        == scenarios.load_fleet("1drone").agents[0].model_copy(update={"id": "x"})
        for a in thirds.agents
    } == {True}
    staggered = scenarios.load_fleet("2drones_staggered")
    assert staggered.agents == scenarios.load_fleet("2drones").agents  # only the offsets differ
    assert staggered.charge_policy.stagger_offsets_s == {"d1": 1800.0}  # half the 3600 s cycle
    speeds = {n: scenarios.load_tactic(n).speed_mps for n in scenarios.names("tactic")}
    assert speeds == {"walk": 1.4, "jog": 2.5, "sprint": scenarios.SPRINT_SPEED_MPS}
    assert adapt.sensor_fov_deg(curves, "drone_cam") == 360.0
    assert adapt.sensor_footprint_radius_m(curves, "drone_cam") == 30.0
    assert adapt.sensor_fov_deg(curves, "ground_cam") == 70.0
    assert adapt.sensor_fov_deg(curves, "fixed_cam") == 60.0
    assert adapt.sensor_footprint_radius_m(curves, "fixed_cam") == 40.0
    benign = {r.cls for r in adapt.benign_routes(site)}
    assert all(
        benign <= adapt.fp_classes(curves, s) for s in ("drone_cam", "ground_cam", "fixed_cam")
    )


def test_unknown_names_raise_clear_errors() -> None:
    with pytest.raises(KeyError, match="moon_base"):
        scenarios.load_site("moon_base")
    with pytest.raises(KeyError, match="fleet_99drones.json"):
        scenarios.load_fleet("99drones")


def test_conftest_fleet_matches_the_scenario_files(fleet_of) -> None:  # type: ignore[no-untyped-def]
    assert fleet_of(1) == scenarios.load_fleet("1drone")
    assert fleet_of(2) == scenarios.load_fleet("2drones")
    assert fleet_of(4) == scenarios.load_fleet("4drones")
