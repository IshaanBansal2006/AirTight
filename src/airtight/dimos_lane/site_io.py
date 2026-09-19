"""Load the frozen example site and fleet from the contracts package."""

from __future__ import annotations

from importlib import resources

from airtight.contracts import FleetConfig, SensorCurves, Site

EXAMPLES = resources.files("airtight.contracts.examples")


def load_example_site() -> Site:
    return Site.model_validate_json(EXAMPLES.joinpath("site.json").read_text())


def load_example_fleet() -> FleetConfig:
    return FleetConfig.model_validate_json(EXAMPLES.joinpath("fleet_config.json").read_text())


def load_stub_sensor_curves() -> SensorCurves:
    return SensorCurves.model_validate_json(EXAMPLES.joinpath("sensor_curve.json").read_text())
