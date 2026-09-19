"""Shared sim fixtures. The yard_night scenario files are the single source of truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from airtight.sim import scenarios

if TYPE_CHECKING:
    from collections.abc import Callable

    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic


def make_fleet(n: int, speed_mps: float | None = None) -> FleetConfig:
    """n copies of the scenario's drone, ids d0..d(n-1); speed_mps overrides the file's speed."""
    template = scenarios.load_fleet("1drone")
    drone = template.agents[0]
    if speed_mps is not None:
        drone = drone.model_copy(update={"speed_mps": speed_mps})
    agents = [drone.model_copy(update={"id": f"d{i}"}) for i in range(n)]
    name = "1drone" if n == 1 else f"{n}drones"
    return template.model_copy(update={"name": name, "agents": agents})


@pytest.fixture
def yard_site() -> Site:
    return scenarios.load_site()


@pytest.fixture
def yard_curve() -> SensorCurves:
    return scenarios.load_sensor_curves()


@pytest.fixture
def yard_tactic() -> Tactic:
    return scenarios.load_tactic("jog")


@pytest.fixture
def fleet_of() -> Callable[..., FleetConfig]:
    return make_fleet
