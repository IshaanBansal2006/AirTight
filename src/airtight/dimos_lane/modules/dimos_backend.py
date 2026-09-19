"""DroneBackend implementation: Go2 goto/pose against dimOS, kinematics otherwise."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from dimos.agents.annotation import skill
from dimos.core.module import Module

if TYPE_CHECKING:
    from numpy.typing import NDArray

MoveTo = Callable[[float, float], str]


class DimosBackend:
    """`goto` / `pose` / `step` matching the frozen DroneBackend protocol.

    The Go2 is dispatched through an injected `move_to(x, y)` (the dimOS skill)
    when present. `step` integrates other agents kinematically; the live Go2
    sim ticks itself, so step is a no-op for that id while a live hook is set.
    """

    GO2_ID = "go2_1"

    def __init__(self, move_to: MoveTo | None = None) -> None:
        from airtight.dimos_lane.modules.sim_fleet import default_drones

        self._move_to = move_to
        self._poses: dict[str, NDArray[np.float64]] = {
            d.drone_id: np.asarray(d.position, dtype=np.float64) for d in default_drones()
        }
        self._speeds: dict[str, float] = {d.drone_id: d.speed for d in default_drones()}
        self._targets: dict[str, NDArray[np.float64]] = {}
        self.gotos: list[tuple[str, list[float]]] = []

    def goto(self, drone_id: str, waypoint: np.ndarray) -> None:
        target = np.asarray(waypoint, dtype=np.float64).reshape(-1)
        if target.size < 2:
            raise ValueError(f"waypoint needs x,y got {target}")
        if target.size == 2:
            z = float(self._poses.get(drone_id, np.zeros(3))[2]) if drone_id in self._poses else 0.0
            target = np.array([target[0], target[1], z], dtype=np.float64)
        else:
            target = target[:3]
        self._targets[drone_id] = target
        self.gotos.append((drone_id, target.tolist()))
        if drone_id == self.GO2_ID and self._move_to is not None:
            self._move_to(float(target[0]), float(target[1]))

    def pose(self, drone_id: str) -> np.ndarray:
        if drone_id not in self._poses:
            self._poses[drone_id] = np.zeros(3, dtype=np.float64)
        return self._poses[drone_id].copy()

    def step(self, dt: float) -> None:
        for drone_id, target in list(self._targets.items()):
            if drone_id == self.GO2_ID and self._move_to is not None:
                continue
            pos = self._poses.setdefault(drone_id, np.zeros(3, dtype=np.float64))
            delta = target - pos
            dist = float(np.linalg.norm(delta))
            if dist < 0.3:
                self._poses[drone_id] = target
                continue
            speed = self._speeds.get(drone_id, 2.0)
            self._poses[drone_id] = pos + delta / dist * min(speed * dt, dist)

    def bind_move_to(self, move_to: MoveTo | None) -> None:
        self._move_to = move_to

    def set_pose(self, drone_id: str, xyz: list[float]) -> None:
        self._poses[drone_id] = np.asarray(xyz, dtype=np.float64)


class DimosBackendModule(Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.inner = DimosBackend()

    @skill
    def goto_xy(self, drone_id: str, x: float, y: float) -> str:
        """Command an agent to a world-frame waypoint."""
        self.inner.goto(drone_id, np.array([x, y], dtype=np.float64))
        return f"{drone_id} -> ({x:.1f},{y:.1f})"

    @skill
    def pose_of(self, drone_id: str) -> str:
        """Return the last known pose of an agent."""
        pos = self.inner.pose(drone_id)
        return f"{drone_id} @ {pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}"
