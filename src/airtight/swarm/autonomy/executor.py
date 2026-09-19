"""Task execution: BT leaves -> DroneBackend (decisions 020 + 008 fidelity ladder).

`DroneBackend` is THE dynamics seam from the fidelity ladder:
`KinematicBackend` (here) integrates simple constant-speed motion in-process;
the future quadrotor/PX4-style backend implements the same three methods
against real dynamics — nothing above this interface changes when fidelity
upgrades (that is the entire point of the seam).

The Executor turns each drone's allocated task list (CBBA path order) into a
queue of behavior trees and ticks them: goto/patrol/sweep tasks become
Sequences of goto leaves (SUCCESS on arrival); follow_track becomes a chase
leaf that reads the live track picture each tick and only ever returns RUNNING
(until preempted) or FAILURE (track lost -> replanner trigger).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from airtight.swarm.autonomy.bt import Action, Node, Sequence, Status
from airtight.swarm.edge import rotation

if TYPE_CHECKING:
    from airtight.swarm.autonomy.decomposer import Task
    from airtight.swarm.autonomy.world_state import DroneState, WorldState

log = logging.getLogger(__name__)

ARRIVE_TOL = 0.3  # m — waypoint arrival radius
FOLLOW_STANDOFF = 2.0  # m above target a follower holds
MIN_APPROACH_SPEED = 0.5  # m/s — SmoothBackend never brakes below this short of a waypoint
HEADING_SPEED_EPS = 0.05  # m/s — below this planar speed the heading is held, not updated


class DroneBackend(Protocol):
    """The 008 dynamics seam. Kinematic today; quadrotor later. Three methods."""

    def goto(self, drone_id: str, waypoint: np.ndarray) -> None: ...
    def pose(self, drone_id: str) -> np.ndarray: ...
    def step(self, dt: float) -> None: ...


class KinematicBackend:
    """Constant-speed point-mass motion toward the latest goto target.

    Positions are mutated on the SHARED DroneState objects (the world state is
    the single source of truth; the Isaac scene renders these same poses).
    """

    def __init__(self, world: WorldState) -> None:
        self.world = world
        self._targets: dict[str, np.ndarray] = {}

    def goto(self, drone_id: str, waypoint: np.ndarray) -> None:
        self._targets[drone_id] = np.asarray(waypoint, dtype=float)

    def pose(self, drone_id: str) -> np.ndarray:
        return np.asarray(self.world.drones[drone_id].position, dtype=float)

    def step(self, dt: float) -> None:
        for drone_id, target in self._targets.items():
            drone = self.world.drones[drone_id]
            pos = np.asarray(drone.position, dtype=float)
            delta = target - pos
            dist = float(np.linalg.norm(delta))
            step = drone.speed * dt
            if dist <= step:  # arrive (snap, zero velocity)
                drone.position = target.tolist()
                drone.velocity = [0.0, 0.0, 0.0]
            else:
                vel = delta / dist * drone.speed
                drone.position = (pos + vel * dt).tolist()
                drone.velocity = vel.tolist()


class SmoothBackend:
    """Acceleration- and yaw-rate-limited point-mass motion.

    Same `DroneBackend` contract as `KinematicBackend`, two differences that
    matter once an IMU is synthesised from the trajectory: velocity changes are
    bounded by `max_accel`, so finite-difference accelerations are finite at
    waypoints; and the platform has an attitude — level, heading along its
    velocity, turning no faster than `max_yaw_rate` — written to
    `DroneState.orientation` so a body-mounted camera has a direction.

    Approach speed is capped at sqrt(2 a d) (the braking distance for the
    remaining distance d) so the drone decelerates into the waypoint rather
    than overshooting it, with a floor so it never stalls short of ARRIVE_TOL.
    """

    def __init__(
        self, world: WorldState, max_accel: float = 2.0, max_yaw_rate: float = 1.5
    ) -> None:
        self.world = world
        self.max_accel = max_accel
        self.max_yaw_rate = max_yaw_rate
        self._targets: dict[str, np.ndarray] = {}
        self._yaw: dict[str, float] = {}

    def goto(self, drone_id: str, waypoint: np.ndarray) -> None:
        self._targets[drone_id] = np.asarray(waypoint, dtype=float)

    def pose(self, drone_id: str) -> np.ndarray:
        return np.asarray(self.world.drones[drone_id].position, dtype=float)

    def step(self, dt: float) -> None:
        for drone_id, drone in self.world.drones.items():
            pos = np.asarray(drone.position, dtype=float)
            vel = np.asarray(drone.velocity, dtype=float)
            target = self._targets.get(drone_id)
            if target is None:
                v_des = np.zeros(3)
            else:
                delta = target - pos
                dist = float(np.linalg.norm(delta))
                if dist < 1e-9:
                    v_des = np.zeros(3)
                else:
                    approach = max(np.sqrt(2.0 * self.max_accel * dist), MIN_APPROACH_SPEED)
                    v_des = delta / dist * min(drone.speed, approach)
            dv = v_des - vel
            dv_max = self.max_accel * dt
            if float(np.linalg.norm(dv)) > dv_max:
                dv *= dv_max / float(np.linalg.norm(dv))
            vel = vel + dv
            step = vel * dt
            if target is not None and float(np.linalg.norm(step)) >= float(
                np.linalg.norm(target - pos)
            ):
                pos, vel = target.copy(), np.zeros(3)  # arrive: snap, stop
            else:
                pos = pos + step
            drone.position = pos.tolist()
            drone.velocity = vel.tolist()
            drone.orientation = rotation.from_yaw(self._turn(drone_id, drone, vel, dt)).tolist()

    def _turn(self, drone_id: str, drone: DroneState, vel: np.ndarray, dt: float) -> float:
        yaw = self._yaw.get(drone_id)
        if yaw is None:
            yaw = rotation.yaw_of(np.asarray(drone.orientation, dtype=float))
        if float(np.hypot(vel[0], vel[1])) > HEADING_SPEED_EPS:
            err = float(rotation.wrap_angle(np.arctan2(vel[1], vel[0]) - yaw))
            yaw += float(np.clip(err, -self.max_yaw_rate * dt, self.max_yaw_rate * dt))
        self._yaw[drone_id] = float(rotation.wrap_angle(yaw))
        return self._yaw[drone_id]


# ---------------------------------------------------------------- tree factory
def _goto_leaf(backend: DroneBackend, drone_id: str, wp: list[float]) -> Action:
    target = np.asarray(wp, dtype=float)

    def fn(ctx: dict) -> Status:
        backend.goto(drone_id, target)
        if float(np.linalg.norm(backend.pose(drone_id) - target)) < ARRIVE_TOL:
            return Status.SUCCESS
        return Status.RUNNING

    return Action(f"goto({drone_id})", fn)


def _follow_leaf(backend: DroneBackend, drone_id: str, task: Task, world: WorldState) -> Action:
    def fn(ctx: dict) -> Status:
        track = world.tracks.get(task.target_track_id)
        if track is None:
            return Status.FAILURE  # track lost -> surfaces as a replan trigger
        goal = np.asarray(track.position, dtype=float) + [0.0, 0.0, FOLLOW_STANDOFF]
        backend.goto(drone_id, goal)
        return Status.RUNNING  # following never "finishes" — it is preempted

    return Action(f"follow({drone_id}->{task.target_track_id})", fn)


def build_tree(task: Task, drone_id: str, backend: DroneBackend, world: WorldState) -> Node:
    if task.task_type in ("goto_waypoint", "patrol_leg", "sweep_cell"):
        return Sequence(*[_goto_leaf(backend, drone_id, wp) for wp in task.waypoints])
    if task.task_type == "follow_track":
        return _follow_leaf(backend, drone_id, task, world)
    raise ValueError(f"no behavior tree for task_type '{task.task_type}'")


# -------------------------------------------------------------------- executor
@dataclass
class _DroneQueue:
    tasks: list[Task] = field(default_factory=list)
    tree: Node | None = None
    current: Task | None = None


class Executor:
    """Ticks per-drone task queues; reports task completions and failures."""

    def __init__(self, backend: DroneBackend, world: WorldState) -> None:
        self.backend = backend
        self.world = world
        self._queues: dict[str, _DroneQueue] = {}
        self.completed: list[str] = []  # task_ids
        self.failed: list[str] = []

    def assign(self, allocation: dict[str, list[Task]]) -> None:
        """Replace all queues (a replan preempts everything — the
        reactive stance: trees are cheap, rebuild rather than patch)."""
        self._queues = {d: _DroneQueue(tasks=list(ts)) for d, ts in allocation.items()}

    def tick(self, dt: float) -> dict[str, Status]:
        """One control cycle: advance dynamics, tick each drone's current tree."""
        self.backend.step(dt)
        statuses: dict[str, Status] = {}
        for drone_id, q in self._queues.items():
            if q.tree is None:
                if not q.tasks:
                    statuses[drone_id] = Status.SUCCESS  # idle
                    continue
                q.current = q.tasks.pop(0)
                q.tree = build_tree(q.current, drone_id, self.backend, self.world)
            status = q.tree.tick({})
            if status == Status.SUCCESS:
                self.completed.append(q.current.task_id)
                q.tree, q.current = None, None
            elif status == Status.FAILURE:
                log.warning("task %s failed on %s", q.current.task_id, drone_id)
                self.failed.append(q.current.task_id)
                q.tree, q.current = None, None
            statuses[drone_id] = status
        return statuses

    def idle(self) -> bool:
        return all(q.tree is None and not q.tasks for q in self._queues.values())
