"""Simulator-free flight harness shared by the SLAMMOT tests: a drone flies the
road scene on the SmoothBackend, an IMU is synthesised at the IMU rate, and at
each fusion frame the mounted camera and the ground radar report detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.autonomy.executor import SmoothBackend
from airtight.swarm.autonomy.world_state import DroneState, WorldState
from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import ImuConfig
from airtight.swarm.edge.imu import ImuSample, ImuSynthesizer
from airtight.swarm.edge.pipeline import road_config
from airtight.swarm.edge.sensing import MeasurementSynthesizer, PlatformTruth, TargetTruth
from airtight.swarm.scene import Scene, demo_scene

if TYPE_CHECKING:
    from collections.abc import Iterator

    from airtight.swarm.edge.observation import CameraModel
    from airtight.swarm.edge.types import Detection

FUSION_DT = 0.1


@dataclass
class Frame:
    t: float
    imu: dict[str, list[ImuSample]]  # samples since the previous frame, per drone
    detections: list[Detection]
    cameras: dict[str, CameraModel]  # TRUE camera models (what the synthesiser used)
    truth: dict[str, DroneState]
    target: TargetTruth


class RoadFlight:
    """Fly `waypoints` per drone; yield one Frame per fusion cycle."""

    def __init__(
        self,
        seed: int = 0,
        drone_ids: list[str] | None = None,
        imu_cfg: ImuConfig | None = None,
        scene: Scene | None = None,
        speed: float = 3.0,
        p_detect: float = 1.0,
        target_moving: bool = True,
    ) -> None:
        self.scene = scene or demo_scene()
        self.drone_ids = drone_ids or ["d0"]
        self.imu_cfg = imu_cfg or ImuConfig()
        self.cfg = road_config(self.scene, self.drone_ids)
        self.rng = np.random.default_rng(seed)
        self.world = WorldState()
        for did in self.drone_ids:
            start = self.scene.fleet[did]
            self.world.update_drone(
                DroneState(
                    drone_id=did,
                    position=list(start.position),
                    orientation=rotation.from_yaw(start.yaw).tolist(),
                    speed=speed,
                )
            )
        self.backend = SmoothBackend(self.world)
        self.imus = {
            did: ImuSynthesizer(self.imu_cfg, np.random.default_rng(seed + 1 + i))
            for i, did in enumerate(self.drone_ids)
        }
        self.synth = MeasurementSynthesizer(self.scene, self.cfg, self.rng, p_detect=p_detect)
        self.target_moving = target_moving

    def target_at(self, t: float) -> TargetTruth:
        v = np.asarray(self.scene.target_velocity) if self.target_moving else np.zeros(3)
        p = np.asarray(self.scene.target_start) + v * t
        return TargetTruth(
            np.concatenate([p, v, self.scene.target_extent]),
            float(np.arctan2(v[1], v[0])) if np.linalg.norm(v[:2]) > 1e-3 else 0.0,
            self.scene.target_color,
        )

    def frames(self, waypoints: dict[str, list[np.ndarray]], duration_s: float) -> Iterator[Frame]:
        dt = self.imu_cfg.dt
        steps_per_frame = int(round(FUSION_DT / dt))
        leg = dict.fromkeys(self.drone_ids, 0)
        for did in self.drone_ids:
            self.backend.goto(did, waypoints[did][0])
            d = self.world.drones[did]
            self.imus[did].sample(0.0, np.asarray(d.velocity), np.asarray(d.orientation))
        t = 0.0
        n_frames = int(round(duration_s / FUSION_DT))
        for _ in range(n_frames):
            imu_batch: dict[str, list[ImuSample]] = {did: [] for did in self.drone_ids}
            for _ in range(steps_per_frame):
                self.backend.step(dt)
                t += dt
                for did in self.drone_ids:
                    d = self.world.drones[did]
                    s = self.imus[did].sample(t, np.asarray(d.velocity), np.asarray(d.orientation))
                    assert s is not None
                    imu_batch[did].append(s)
                    wps = waypoints[did]
                    if (
                        leg[did] < len(wps) - 1
                        and np.linalg.norm(np.asarray(d.position) - wps[leg[did]]) < 0.4
                    ):
                        leg[did] += 1
                        self.backend.goto(did, wps[leg[did]])
            platforms = {
                did: PlatformTruth(
                    np.asarray(self.world.drones[did].position),
                    np.asarray(self.world.drones[did].orientation),
                )
                for did in self.drone_ids
            }
            target = self.target_at(t)
            dets, cams = self.synth.observe(t, platforms, [target])
            yield Frame(
                t=t,
                imu=imu_batch,
                detections=dets,
                cameras=cams,
                truth={did: self.world.drones[did] for did in self.drone_ids},
                target=target,
            )


def road_waypoints(x_end: float = 65.0, y: float = 3.0, z: float = 4.0) -> list[np.ndarray]:
    return [np.array([x_end, y, z])]
