from __future__ import annotations

from importlib import resources

import pytest

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic

EXAMPLES = resources.files("airtight.contracts.examples")


@pytest.fixture(scope="session")
def site() -> Site:
    return Site.model_validate_json(EXAMPLES.joinpath("site.json").read_text())


@pytest.fixture(scope="session")
def fleet() -> FleetConfig:
    return FleetConfig.model_validate_json(EXAMPLES.joinpath("fleet_config.json").read_text())


@pytest.fixture(scope="session")
def curves() -> SensorCurves:
    return SensorCurves.model_validate_json(EXAMPLES.joinpath("sensor_curve.json").read_text())


@pytest.fixture(scope="session")
def hand_tactics() -> list[Tactic]:
    return [
        Tactic.model_validate_json(EXAMPLES.joinpath(n).read_text())
        for n in ("tactic.json", "tactic_decoy.json", "tactic_blind_spot.json")
    ]
