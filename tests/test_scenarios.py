from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from airtight.contracts import FleetConfig, SensorCurves, Site
from airtight.redteam import RedTeamConfig, sample_tactic, validate
from airtight.redteam.families import FAMILIES

SCEN = Path(__file__).parents[1] / "scenarios" / "logistics_yard"
CURVES = SensorCurves.model_validate_json(
    resources.files("airtight.contracts.examples").joinpath("sensor_curve.json").read_text()
)


@pytest.fixture(scope="module")
def site() -> Site:
    return Site.model_validate_json((SCEN / "site.json").read_text())


@pytest.fixture(scope="module")
def cfg() -> RedTeamConfig:
    return RedTeamConfig.model_validate_json((SCEN / "redteam_config.json").read_text())


def _fleets() -> list[Path]:
    return sorted(p for p in (SCEN / "fleets").glob("*.json") if p.name != "sweep.json")


def test_sweep_lists_every_fleet_and_the_baseline() -> None:
    sweep = json.loads((SCEN / "fleets" / "sweep.json").read_text())
    names = {p.stem for p in _fleets()}
    assert set(sweep["configs"]) == names and sweep["baseline"] in names and len(names) == 12


@pytest.mark.parametrize("fleet_path", _fleets(), ids=lambda p: p.stem)
def test_fleet_configs_validate_and_are_priced(fleet_path: Path) -> None:
    fleet = FleetConfig.model_validate_json(fleet_path.read_text())
    assert fleet.name == fleet_path.stem
    assert {a.sensor_type for a in fleet.agents} <= set(CURVES.curves)
    assert fleet.cost_per_hour() > 0
    if fleet.charge_policy.stagger_offsets_s:
        assert set(fleet.charge_policy.stagger_offsets_s) <= {a.id for a in fleet.agents}


def test_site_admits_every_family(site: Site, cfg: RedTeamConfig) -> None:
    fleet = FleetConfig.model_validate_json(
        (SCEN / "fleets" / "d2_go2_guard_sync.json").read_text()
    )
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        for _ in range(5):
            t = sample_tactic(fam, site, fleet, CURVES, rng, cfg)
            assert validate(t, site, cfg) == []
    assert {s.sensor_type for s in site.fixed_sensors} <= set(CURVES.curves)
    assert {r.cls for r in site.benign_routes} <= set.intersection(
        *(set(c.pfa_per_look_by_class) for c in CURVES.curves.values())
    )
