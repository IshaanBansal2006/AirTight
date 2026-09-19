"""Lane-local scenarios: directories of JSON files that validate against the contract models.

A scenario directory holds site.json, sensor_curve.json, fleet_<name>.json files and
tactic_<name>.json files. These are lane B's working scenarios, not contract examples.

SPRINT_SPEED_MPS is a lane-local stand-in: the contract puts no upper bound on Tactic.speed_mps
and lane C's validator takes its cap as an argument, so no team-wide cap is written down yet.
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from pydantic import BaseModel

SPRINT_SPEED_MPS = 6.0  # the team has not fixed a speed cap yet; change it here when it does
DEFAULT_SCENARIO = "yard_night"


def scenario_dir(scenario: str = DEFAULT_SCENARIO) -> Traversable:
    root = resources.files(__name__).joinpath(scenario)
    if not root.is_dir():
        raise KeyError(f"no scenario {scenario!r}; known: {scenario_names()}")
    return root


def scenario_names() -> list[str]:
    return sorted(
        p.name for p in resources.files(__name__).iterdir() if p.is_dir() and p.name[0] != "_"
    )


def names(prefix: str, scenario: str = DEFAULT_SCENARIO) -> list[str]:
    """The <name> part of every <prefix>_<name>.json in the scenario, sorted."""
    start, end = f"{prefix}_", ".json"
    return sorted(
        p.name[len(start) : -len(end)]
        for p in scenario_dir(scenario).iterdir()
        if p.name.startswith(start) and p.name.endswith(end)
    )


def _load[M: BaseModel](model: type[M], filename: str, scenario: str) -> M:
    path = scenario_dir(scenario).joinpath(filename)
    if not path.is_file():
        raise KeyError(f"scenario {scenario!r} has no file {filename!r}")
    return model.model_validate_json(path.read_text())


def load_site(scenario: str = DEFAULT_SCENARIO) -> Site:
    return _load(Site, "site.json", scenario)


def load_sensor_curves(scenario: str = DEFAULT_SCENARIO) -> SensorCurves:
    return _load(SensorCurves, "sensor_curve.json", scenario)


def load_fleet(name: str, scenario: str = DEFAULT_SCENARIO) -> FleetConfig:
    return _load(FleetConfig, f"fleet_{name}.json", scenario)


def load_tactic(name: str, scenario: str = DEFAULT_SCENARIO) -> Tactic:
    return _load(Tactic, f"tactic_{name}.json", scenario)
