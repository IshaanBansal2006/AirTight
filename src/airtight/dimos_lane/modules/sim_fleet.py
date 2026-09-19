"""Kinematic drone markers. Topics are prefixed by robot id. No second physics world."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dimos.agents.annotation import skill
from dimos.core.module import Module

from airtight.dimos_lane.modules.allocator import drone_state_for
from airtight.dimos_lane.site_io import load_example_fleet, load_example_site

if TYPE_CHECKING:
    from airtight.contracts.fleet import FleetConfig
    from airtight.swarm.autonomy.world_state import DroneState


def _default_pose(agent_id: str, agent_type: str) -> list[float]:
    site = load_example_site()
    if agent_type == "go2" and site.docks:
        dock = site.docks[0].position
        return [dock.x, dock.y, 0.0]
    if agent_type == "guard":
        gate = site.entry_points[0].position
        return [gate.x, gate.y, 0.0]
    # drones hold a patrol altitude above the asset
    return [site.asset.x + (hash(agent_id) % 13) - 6, site.asset.y, 8.0]


def default_drones(fleet: FleetConfig | None = None) -> list[DroneState]:
    fleet = fleet or load_example_fleet()
    return [
        drone_state_for(
            agent.id,
            agent.type,
            _default_pose(agent.id, agent.type),
            agent.speed_mps,
        )
        for agent in fleet.agents
    ]


def topic_for(robot_id: str, leaf: str = "pose") -> str:
    return f"/{robot_id}/{leaf}"


class SimFleet:
    def __init__(self, fleet: FleetConfig | None = None) -> None:
        self.states = {d.drone_id: d for d in default_drones(fleet)}

    def set_pose(self, robot_id: str, x: float, y: float, z: float = 0.0) -> None:
        drone = self.states.get(robot_id)
        if drone is None:
            drone = drone_state_for(robot_id, "drone", [x, y, z], 8.0)
            self.states[robot_id] = drone
        drone.position = [x, y, z]

    def snapshot(self) -> dict[str, list[float]]:
        return {did: list(d.position) for did, d in self.states.items()}

    def drones(self) -> list[DroneState]:
        return list(self.states.values())


class SimFleetModule(Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.inner = SimFleet()

    @skill
    def set_marker(self, robot_id: str, x: float, y: float, z: float = 8.0) -> str:
        """Place a kinematic drone marker. Topic is /{robot_id}/pose."""
        self.inner.set_pose(robot_id, x, y, z)
        return f"{topic_for(robot_id)} <- ({x:.1f},{y:.1f},{z:.1f})"

    @skill
    def fleet_markers(self) -> str:
        """Return every marker pose."""
        parts = [
            f"{did}@{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
            for did, pos in self.inner.snapshot().items()
        ]
        return "; ".join(parts)
