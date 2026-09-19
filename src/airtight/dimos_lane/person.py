"""Intruder / person poses for MuJoCo (`/person_pose`) and offline replay."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

from airtight.contracts.site import XY

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airtight.contracts.site import Site

INTRUDER_RANGE_M = 6.0
INTRUDER_BEARING_DEG = 0.0
INTRUDER_YAW_RAD = pi  # face the Go2, which spawns looking +x


def interpolate_path(
    waypoints: Sequence[XY], speed_mps: float, dt: float
) -> list[tuple[float, XY]]:
    """Sample a polyline at `dt` seconds. First sample is t=0 at waypoints[0]."""
    if len(waypoints) == 0:
        return []
    if speed_mps <= 0:
        raise ValueError(f"speed_mps must be positive, got {speed_mps}")
    samples: list[tuple[float, XY]] = [(0.0, waypoints[0])]
    t = 0.0
    x, y = waypoints[0].x, waypoints[0].y
    for nxt in waypoints[1:]:
        while True:
            dx, dy = nxt.x - x, nxt.y - y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1e-9:
                break
            step = speed_mps * dt
            if step >= dist:
                t += dist / speed_mps
                x, y = nxt.x, nxt.y
                samples.append((t, XY(x=x, y=y)))
                break
            x += dx / dist * step
            y += dy / dist * step
            t += dt
            samples.append((t, XY(x=x, y=y)))
    return samples


def go2_spawn_xy(site: Site | None = None) -> XY:
    """Dock A on the example yard: Go2 start, camera looking +x."""
    if site is None:
        from airtight.dimos_lane.site_io import load_example_site

        site = load_example_site()
    if site.docks:
        return site.docks[0].position
    return XY(x=10.0, y=70.0)


def mujoco_start_pos(site: Site | None = None) -> str:
    spawn = go2_spawn_xy(site)
    return f"{spawn.x:.1f},{spawn.y:.1f}"


def default_intruder_xy(site: Site | None = None) -> XY:
    """Person standing INTRUDER_RANGE_M metres in front of the Go2 (+x)."""
    from airtight.dimos_lane.calibration import pose_for_look

    spawn = go2_spawn_xy(site)
    x, y = pose_for_look(INTRUDER_RANGE_M, INTRUDER_BEARING_DEG, (spawn.x, spawn.y))
    return XY(x=x, y=y)


def publish_person_pose(x: float, y: float, z: float = 0.0, yaw_rad: float = 0.0) -> None:
    """Broadcast a pose on dimOS `/person_pose`. No-op if transports are unavailable."""
    from math import cos, sin

    from dimos.core.transport_factory import make_transport
    from dimos.msgs.geometry_msgs.Pose import Pose

    transport = make_transport("/person_pose", Pose)
    pose = Pose(
        position=[x, y, z],
        orientation=[0.0, 0.0, sin(yaw_rad / 2), cos(yaw_rad / 2)],
    )
    transport.broadcast(None, pose)
    transport.stop()


def place_intruder(site: Site | None = None) -> XY:
    """Put the MuJoCo person mocap in front of the Go2 camera."""
    xy = default_intruder_xy(site)
    publish_person_pose(xy.x, xy.y, z=0.0, yaw_rad=INTRUDER_YAW_RAD)
    return xy
