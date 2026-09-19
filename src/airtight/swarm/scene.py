"""The demo scene, shared by the simulator, the sensing harness and the tests.

One description of what is in the world — prior-mapped road signs, parked
vehicles of several sizes and colours, the fleet's start poses, the moving
target, the ground radar — so the Isaac scene renders exactly what the
measurement synthesiser observes and the ego filter's prior map matches the
signs actually placed (decisions 017, 018).

Pure dataclasses + numpy on purpose: the Isaac scene script runs under the
simulator's bundled Python, which has neither pydantic nor this package
installed, and imports this module by path.

World frame: z up, ground at z = 0. The road runs along +x at y = 0. Signs
stand on the verges facing along the road (yaw 0: thin along x, wide along y),
so a drone flying down the road sees them face-on. Vehicles park on the
shoulders, aligned with the road.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Standard sign faces (m), as [thickness, width, height] in the sign's own frame.
# Standard size is what makes a sign a range-giving landmark: apparent width
# in pixels + known width in metres -> distance.
SIGN_EXTENTS: dict[str, tuple[float, float, float]] = {
    "stop": (0.05, 0.75, 0.75),
    "speed_limit": (0.05, 0.60, 0.75),
}
SIGN_CENTER_HEIGHT = 2.1  # m — centre of the face above the ground

# Vehicle bodies (m), as [length, width, height]. NOT standard: same class,
# different sizes, so apparent size cannot give range — vehicles are
# bearing-only landmarks until seen from two viewpoints.
VEHICLE_EXTENTS: dict[str, tuple[float, float, float]] = {
    "compact": (3.8, 1.7, 1.5),
    "sedan": (4.6, 1.8, 1.45),
    "van": (5.2, 2.0, 2.2),
    "pickup": (5.5, 2.0, 1.9),
}


@dataclass(frozen=True)
class Sign:
    sign_id: str
    sign_type: str
    position: tuple[float, float, float]
    yaw: float = 0.0

    @property
    def extent(self) -> NDArray[np.float64]:
        return np.asarray(SIGN_EXTENTS[self.sign_type], dtype=float)


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: str
    body: str
    color: str
    position: tuple[float, float, float]
    yaw: float = 0.0

    @property
    def extent(self) -> NDArray[np.float64]:
        return np.asarray(VEHICLE_EXTENTS[self.body], dtype=float)


@dataclass(frozen=True)
class DroneStart:
    position: tuple[float, float, float]
    yaw: float = 0.0


@dataclass(frozen=True)
class Scene:
    signs: tuple[Sign, ...]
    vehicles: tuple[Vehicle, ...]
    fleet: dict[str, DroneStart]
    radar_position: tuple[float, float, float]
    target_start: tuple[float, float, float]
    target_velocity: tuple[float, float, float]
    target_extent: tuple[float, float, float]
    target_color: str = "silver"
    road_y_halfwidth: float = 1.5
    patrol_area: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def sign(self, sign_id: str) -> Sign:
        for s in self.signs:
            if s.sign_id == sign_id:
                return s
        raise KeyError(f"no sign '{sign_id}' in the scene")


def _parked(vid: str, body: str, color: str, x: float, side: float) -> Vehicle:
    height = VEHICLE_EXTENTS[body][2]
    return Vehicle(vid, body, color, (x, side * 2.6, height / 2.0))


def demo_scene() -> Scene:
    """The road-patrol scene: five signs, four parked vehicles, one moving sedan."""
    signs = tuple(
        Sign(f"s{i}", kind, (x, side * 4.0, SIGN_CENTER_HEIGHT))
        for i, (x, kind, side) in enumerate(
            [
                (0.0, "stop", 1.0),
                (15.0, "speed_limit", -1.0),
                (30.0, "stop", 1.0),
                (45.0, "speed_limit", -1.0),
                (60.0, "stop", 1.0),
            ]
        )
    )
    vehicles = (
        _parked("v0", "compact", "blue", 8.0, 1.0),
        _parked("v1", "sedan", "white", 22.0, -1.0),
        _parked("v2", "van", "red", 37.0, 1.0),
        _parked("v3", "pickup", "black", 52.0, -1.0),
    )
    return Scene(
        signs=signs,
        vehicles=vehicles,
        fleet={"d0": DroneStart((-8.0, 3.0, 4.0)), "d1": DroneStart((-8.0, -3.0, 4.0))},
        radar_position=(-15.0, -10.0, 3.0),
        target_start=(-5.0, 0.0, 0.75),
        target_velocity=(2.0, 0.0, 0.0),
        target_extent=(4.5, 1.8, 1.5),
        patrol_area=((-10.0, -6.0), (70.0, -6.0), (70.0, 6.0), (-10.0, 6.0)),
    )
