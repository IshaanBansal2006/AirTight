"""SLAMMOT pipeline (decisions 017-019): per-drone ego filters + the consider
tracker, and the routing between them that makes it SLAM *and* MOT.

Per fusion cycle:
    1. build a MountedCamera per drone-borne camera from that drone's ego
       ESTIMATE (pose + 6x6 covariance) — the tracker never sees truth;
    2. tracker.step() with every radar return and every camera VEHICLE box:
       radar-initiated tracks gate the moving target's boxes and claim them;
    3. what the tracker did not claim — all sign boxes, and the vehicle boxes
       no track wanted — goes to the owning drone's ego filter: signs as
       prior-map fixes, vehicles as landmark candidates (delayed init + the
       static-consistency check are the ego filter's own defence).

That order is the moving-object discrimination of 018 in one sentence: the
radar's Doppler sees motion the camera cannot, so a moving vehicle becomes a
TRACK before it can become a LANDMARK. The ego filter's bearings-only check
is the second line, for what the radar does not cover.

IMU samples arrive between cycles (`imu()`); the ego filters integrate them
as they come, so at cycle time every pose estimate is current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge.config import CameraConfig, EgoConfig, RadarConfig, TrackerConfig
from airtight.swarm.edge.ego import EgoFilter, UpdateReport
from airtight.swarm.edge.schmidt import ConsiderPose, MountedCamera
from airtight.swarm.edge.tracker import MultiTargetTracker
from airtight.swarm.schemas import DronePoseFrame, DronePoseMsg

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from airtight.swarm.edge.imu import ImuSample
    from airtight.swarm.edge.types import Detection, Track
    from airtight.swarm.scene import Scene


def road_config(scene: Scene, drone_ids: list[str]) -> TrackerConfig:
    """Sensors for the road scene: one mounted camera per drone, the ground
    radar, and a vehicle-class birth prior (the only target is a car)."""
    sensors: dict[str, CameraConfig | RadarConfig] = {
        f"cam_{d}": CameraConfig(platform=d) for d in drone_ids
    }
    sensors["radar_gs"] = RadarConfig(position=list(scene.radar_position))
    return TrackerConfig(
        sensors=sensors,
        birth_extent=[4.5, 1.85, 1.7],  # type: ignore[arg-type]
        birth_extent_std=[1.0, 0.4, 0.5],
    )


@dataclass
class PlatformInit:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    orientation: NDArray[np.float64]


@dataclass
class CycleResult:
    t: float
    tracks: list[Track]
    poses: DronePoseFrame
    reports: dict[str, UpdateReport] = field(default_factory=dict)
    claimed: int = 0


class SlammotPipeline:
    def __init__(
        self,
        config: TrackerConfig,
        ego_cfg: EgoConfig,
        scene: Scene,
        fleet: dict[str, PlatformInit],
    ) -> None:
        self.cfg = config
        self.tracker = MultiTargetTracker(config)
        self.egos = {
            did: EgoFilter(ego_cfg, scene.signs, f.position, f.velocity, f.orientation)
            for did, f in fleet.items()
        }
        self.camera_of: dict[str, str] = {}  # drone_id -> sensor_id
        for sid, s in config.sensors.items():
            if isinstance(s, CameraConfig) and s.platform is not None:
                if s.platform not in self.egos:
                    raise ValueError(
                        f"camera '{sid}' rides '{s.platform}', which is not in the fleet "
                        f"{sorted(self.egos)} — sensors and fleet must name the same drones"
                    )
                self.camera_of[s.platform] = sid
        self._R = {
            sid: np.diag(np.square(np.asarray(s.measurement_noise, dtype=float)))
            for sid, s in config.sensors.items()
        }

    def imu(self, drone_id: str, sample: ImuSample) -> None:
        self.egos[drone_id].predict(sample)

    def _mounted(self) -> dict[str, MountedCamera]:
        out = {}
        for did, sid in self.camera_of.items():
            ego = self.egos[did]
            cfg = self.cfg.sensors[sid]
            assert isinstance(cfg, CameraConfig)
            out[sid] = MountedCamera(
                cfg, ConsiderPose(did, ego.p.copy(), ego.q.copy(), ego.pose_cov())
            )
        return out

    def cycle(self, t: float, detections: list[Detection]) -> CycleResult:
        mounted = self._mounted()
        for_tracker = [d for d in detections if d.class_label != "sign"]
        confirmed = self.tracker.step(for_tracker, camera_models=dict(mounted))
        claimed = {id(d) for d in self.tracker.claimed}

        reports: dict[str, UpdateReport] = {}
        for did, sid in self.camera_of.items():
            mine = [
                d
                for d in detections
                if d.sensor_id == sid and (d.class_label == "sign" or id(d) not in claimed)
            ]
            cfg = self.cfg.sensors[sid]
            assert isinstance(cfg, CameraConfig)
            reports[did] = self.egos[did].update(mine, cfg, t, self._R[sid])

        poses = DronePoseFrame(
            timestamp=t, poses=[ego.to_msg(did, t) for did, ego in self.egos.items()]
        )
        return CycleResult(
            t=t, tracks=confirmed, poses=poses, reports=reports, claimed=len(claimed)
        )

    def pose_msgs(self, t: float) -> list[DronePoseMsg]:
        return [ego.to_msg(did, t) for did, ego in self.egos.items()]
