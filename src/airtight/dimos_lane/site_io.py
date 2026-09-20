"""Load frozen sites and fleets from contracts examples or the logistics-yard scenario."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from airtight.contracts import FleetConfig, SensorCurves, Site

EXAMPLES = resources.files("airtight.contracts.examples")
LOGISTICS = Path(__file__).resolve().parents[3] / "scenarios" / "logistics_yard"


def load_example_site() -> Site:
    return Site.model_validate_json(EXAMPLES.joinpath("site.json").read_text())


def load_example_fleet() -> FleetConfig:
    return FleetConfig.model_validate_json(EXAMPLES.joinpath("fleet_config.json").read_text())


def load_stub_sensor_curves() -> SensorCurves:
    return SensorCurves.model_validate_json(EXAMPLES.joinpath("sensor_curve.json").read_text())


def load_logistics_site() -> Site:
    return Site.model_validate_json((LOGISTICS / "site.json").read_text())


def load_logistics_fleet(name: str) -> FleetConfig:
    return FleetConfig.model_validate_json((LOGISTICS / "fleets" / f"{name}.json").read_text())


def load_logistics_curves() -> SensorCurves:
    return SensorCurves.model_validate_json((LOGISTICS / "sensor_curve.json").read_text())
