"""CBBA allocator wrapped as a dimOS module. Does not edit swarm/."""

from __future__ import annotations

from typing import Any

from dimos.agents.annotation import skill
from dimos.core.module import Module

from airtight.swarm.autonomy.allocator import CBBAAllocator
from airtight.swarm.autonomy.decomposer import Task
from airtight.swarm.autonomy.world_state import DroneState

VERIFY_CAPABILITY = "verify_ground"

CAPABILITIES: dict[str, frozenset[str]] = {
    "drone": frozenset({"camera", "verify_air"}),
    "go2": frozenset({"camera", VERIFY_CAPABILITY}),
    "guard": frozenset({"human", VERIFY_CAPABILITY}),
}


def drone_state_for(
    agent_id: str,
    agent_type: str,
    position: list[float],
    speed: float,
    *,
    available: bool = True,
    battery: float = 1.0,
) -> DroneState:
    return DroneState(
        drone_id=agent_id,
        position=position,
        speed=speed,
        capabilities=CAPABILITIES.get(agent_type, frozenset({"camera"})),
        available=available,
        battery=battery,
    )


def verify_task(task_id: str, x: float, y: float) -> Task:
    return Task(
        task_id=task_id,
        intent_id=task_id,
        task_type="goto_waypoint",
        waypoints=[[x, y, 0.0]],
        required_capability=VERIFY_CAPABILITY,
        priority=1,
        reward=10.0,
    )


class Allocator:
    """In-process CBBA wrapper used by the module and by tests."""

    def __init__(self) -> None:
        self._cbba = CBBAAllocator()
        self.last_assignment: dict[str, list[str]] = {}

    def allocate(self, tasks: list[Task], drones: list[DroneState]) -> dict[str, list[Task]]:
        assignment = self._cbba.allocate(tasks, drones)
        self.last_assignment = {
            drone_id: [task.task_id for task in path] for drone_id, path in assignment.items()
        }
        return assignment

    def winner_for(self, task_id: str) -> str | None:
        for drone_id, task_ids in self.last_assignment.items():
            if task_id in task_ids:
                return drone_id
        return None


class AllocatorModule(Module):
    """dimOS module: one CBBA auction per skill call."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.inner = Allocator()

    @skill
    def allocate_verify(self, x: float, y: float, task_id: str = "verify") -> str:
        """Auction a ground-verify task at (x, y). Returns the winning agent id."""
        from airtight.dimos_lane.modules.sim_fleet import default_drones

        assignment = self.inner.allocate([verify_task(task_id, x, y)], default_drones())
        winner = self.inner.winner_for(task_id)
        paths = {did: [t.task_id for t in path] for did, path in assignment.items()}
        return f"winner={winner} assignment={paths}"
