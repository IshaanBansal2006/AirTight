"""L2 world state — the bookkeeping every autonomy component reads.

Holds what L2 knows: the drone fleet (poses, batteries, capabilities) and the
live track picture from L1 (TrackMsg feed). Deliberately dumb: no inference, no
prediction — estimation belongs to L1, decisions to the decomposer/allocator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from airtight.swarm.schemas import DronePoseFrame, DronePoseMsg, TrackMsg


@dataclass
class DroneState:
    """One platform's status as known to L2.

    `position` / `velocity` / `orientation` are TRUTH, owned by the backend that
    moves the drone. `pose_estimate` is L1's belief about the same platform — what the drone itself would have to navigate by.
    """

    drone_id: str
    position: list[float]  # [x, y, z] m, world frame
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    pose_estimate: DronePoseMsg | None = None  # None until L1 has published one
    battery: float = 1.0  # 0..1
    speed: float = 2.0  # nominal cruise speed m/s (allocator travel-time model)
    capabilities: frozenset[str] = frozenset()  # e.g. {"camera", "radar"}
    available: bool = True


class WorldState:
    """Aggregated L2 picture: fleet + tracks, timestamped by the newest input."""

    def __init__(self) -> None:
        self.drones: dict[str, DroneState] = {}
        self.tracks: dict[int, TrackMsg] = {}
        self.time: float = 0.0

    def update_drone(self, drone: DroneState) -> None:
        self.drones[drone.drone_id] = drone

    def update_tracks(self, tracks: list[TrackMsg], timestamp: float) -> None:
        """Replace the track picture (L1 publishes complete confirmed sets)."""
        self.tracks = {t.track_id: t for t in tracks}
        self.time = max(self.time, timestamp)

    def update_pose_estimates(self, frame: DronePoseFrame) -> None:
        """Attach L1's latest pose estimate to each known drone; unknown ids are
        ignored (a fleet mismatch is an L1/L2 config error, logged by the node)."""
        for est in frame.poses:
            drone = self.drones.get(est.drone_id)
            if drone is not None:
                drone.pose_estimate = est
        self.time = max(self.time, frame.timestamp)

    def available_drones(self) -> list[DroneState]:
        return [d for d in self.drones.values() if d.available and d.battery > 0.1]
