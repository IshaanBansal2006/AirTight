"""Shared sim fixtures, built from contract models. Step 2.6 turns these into scenario files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from airtight.contracts import (
    XY,
    AgentSpec,
    BenignRoute,
    Dock,
    EntryPoint,
    FleetConfig,
    SensorCurve,
    SensorCurves,
    Site,
    Tactic,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _xy(x: float, y: float) -> XY:
    return XY(x=x, y=y)


def make_yard_site() -> Site:
    return Site(
        name="yard-a",
        bounds=(0.0, 0.0, 300.0, 200.0),
        perimeter=[_xy(20, 20), _xy(280, 20), _xy(280, 180), _xy(20, 180)],
        entry_points=[
            EntryPoint(id="west_fence", position=_xy(20, 100), kind="fence_gap"),
            EntryPoint(id="north_gate", position=_xy(150, 180), kind="gate"),
        ],
        asset=_xy(150, 100),
        response_time_s=25.0,
        docks=[
            Dock(id="dock_sw", position=_xy(40, 40)),
            Dock(id="dock_ne", position=_xy(260, 160)),
        ],
        fixed_sensors=[],
        benign_routes=[
            BenignRoute(
                id="fox",
                cls="animal",
                waypoints=[_xy(20, 60), _xy(150, 100), _xy(280, 140)],
                arrival_rate_per_hour=6.0,
            ),
            BenignRoute(
                id="tarp",
                cls="debris",
                waypoints=[_xy(100, 30), _xy(130, 50)],
                arrival_rate_per_hour=4.0,
            ),
        ],
    )


def _curve(sensor_type: str, fov_deg: float, look_rate_hz: float = 2.0) -> SensorCurve:
    return SensorCurve(
        sensor_type=sensor_type,
        range_bins_m=[5, 10, 15, 20, 30],
        pd_per_look=[0.7, 0.6, 0.5, 0.4, 0.0],
        pfa_per_look_by_class={"animal": 0.08, "debris": 0.03},
        fov_deg=fov_deg,
        look_rate_hz=look_rate_hz,
    )


def make_yard_curve(drone_fov_deg: float = 360.0) -> SensorCurves:
    return SensorCurves(
        source="synthetic test curve",
        curves={"cam": _curve("cam", drone_fov_deg), "fixed_cam": _curve("fixed_cam", 90.0)},
    )


def make_fleet(n: int, speed_mps: float = 8.0) -> FleetConfig:
    return FleetConfig(
        name=f"{n}drones",
        agents=[
            AgentSpec(
                id=f"d{i}",
                type="drone",
                speed_mps=speed_mps,
                endurance_s=1500,
                charge_time_s=2100,
                sensor_type="cam",
            )
            for i in range(n)
        ],
        cost_per_hour_by_type={"drone": 6.0},
    )


def make_yard_tactic() -> Tactic:
    return Tactic(
        id="cw-west",
        family="charging_window",
        entry_id="west_fence",
        phase=0.45,
        speed_mps=2.5,
        waypoints=[_xy(150, 100)],
    )


@pytest.fixture
def yard_site() -> Site:
    return make_yard_site()


@pytest.fixture
def yard_curve() -> SensorCurves:
    return make_yard_curve()


@pytest.fixture
def yard_tactic() -> Tactic:
    return make_yard_tactic()


@pytest.fixture
def fleet_of() -> Callable[[int], FleetConfig]:
    return make_fleet
