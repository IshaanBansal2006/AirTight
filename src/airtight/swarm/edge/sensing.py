"""Measurement synthesis for drone-mounted cameras and the ground radar
(decisions 017/018): the harness half of L1, kept out of the ROS node so the
tests can drive it.

Given the TRUE world — drone poses, the moving target, the scene's signs and
parked vehicles — produce what each sensor would report:

    camera on drone d:  a Detection per visible box (target, sign, vehicle),
                        pixel bbox [u, v, w, h] + noise, class label, and the
                        attributes a detector would attach (sign type, colour)
    ground radar:       [range, azimuth, elevation, doppler] of the target

Visibility is `in_view` (017): a box is reported only where h_camera models
it. Signs and vehicles are reported with the SAME class labels a real detector
would give — "sign" and "vehicle" — so telling a parked vehicle from the
moving target is left to the estimator, where it belongs (018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import CameraConfig, CameraMount, RadarConfig, TrackerConfig
from airtight.swarm.edge.observation import (
    CameraModel,
    h_camera_box,
    h_radar,
    in_view_box,
)
from airtight.swarm.edge.types import Detection

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from airtight.swarm.scene import Scene

# Camera axes expressed in the body frame for a level, forward-looking camera:
# camera x (right) = body -y, camera y (down) = body -z, camera z (fwd) = body x.
_R_BC_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def mount_rotation(mount: CameraMount) -> NDArray[np.float64]:
    """R_bc: camera-frame vectors -> body-frame vectors, for this mount.

    Pitch is a rotation about body y (left); positive tilts body x (forward)
    toward body -z (down). Yaw about body z applied after.
    """
    pitch = np.deg2rad(mount.pitch_deg)
    yaw = np.deg2rad(mount.yaw_deg)
    R_pitch = rotation.to_matrix(rotation.exp(np.array([0.0, pitch, 0.0])))
    R_yaw = rotation.to_matrix(rotation.exp(np.array([0.0, 0.0, yaw])))
    return np.asarray(R_yaw @ R_pitch @ _R_BC_LEVEL, dtype=float)


def camera_model(
    position: NDArray[np.float64],
    orientation: NDArray[np.float64],
    cfg: CameraConfig,
) -> CameraModel:
    """Extrinsics of a mounted camera at this platform pose (intrinsics from cfg)."""
    R_wb = rotation.to_matrix(np.asarray(orientation, dtype=float))
    R_wc_world_from_cam = R_wb @ mount_rotation(cfg.mount)
    return CameraModel(
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy,
        width=cfg.width,
        height=cfg.height,
        R_wc=R_wc_world_from_cam.T,  # observation.py wants world -> camera
        t_w=np.asarray(position, dtype=float) + R_wb @ np.asarray(cfg.mount.offset),
    )


@dataclass
class PlatformTruth:
    position: NDArray[np.float64]
    orientation: NDArray[np.float64]


@dataclass
class TargetTruth:
    state: NDArray[np.float64]  # 9-D [p, v, L]
    yaw: float
    color: str


class MeasurementSynthesizer:
    """Truth in, detections out. One instance per run; `rng` makes it reproducible."""

    def __init__(
        self, scene: Scene, config: TrackerConfig, rng: np.random.Generator, p_detect: float = 1.0
    ) -> None:
        self.scene = scene
        self.cfg = config
        self.rng = rng
        self.p_detect = p_detect
        self._R = {
            sid: np.diag(np.square(np.asarray(s.measurement_noise, dtype=float)))
            for sid, s in config.sensors.items()
        }

    # ------------------------------------------------------------ cameras
    def cameras(self, platforms: dict[str, PlatformTruth]) -> dict[str, CameraModel]:
        """Extrinsics for every platform-mounted camera at these TRUE poses."""
        out: dict[str, CameraModel] = {}
        for sid, s in self.cfg.sensors.items():
            if isinstance(s, CameraConfig) and s.platform is not None:
                truth = platforms.get(s.platform)
                if truth is None:
                    raise ValueError(
                        f"camera '{sid}' rides '{s.platform}' but no truth pose was given "
                        f"for it — pass every platform in TrackerConfig.sensors"
                    )
                out[sid] = camera_model(truth.position, truth.orientation, s)
        return out

    def _camera_detection(
        self,
        sid: str,
        t: float,
        center: NDArray[np.float64],
        extent: NDArray[np.float64],
        yaw: float,
        cam: CameraModel,
        label: str,
        conf: float,
        attributes: dict[str, str],
    ) -> Detection | None:
        if not in_view_box(center, extent, yaw, cam) or self.rng.uniform() > self.p_detect:
            return None
        z = h_camera_box(center, extent, yaw, cam)
        return Detection(
            sensor_id=sid,
            timestamp=t,
            measurement=z + self.rng.multivariate_normal(np.zeros(4), self._R[sid]),
            class_label=label,
            class_confidence=conf,
            attributes=dict(attributes),
        )

    def observe(
        self, t: float, platforms: dict[str, PlatformTruth], targets: list[TargetTruth]
    ) -> tuple[list[Detection], dict[str, CameraModel]]:
        """All detections at time t, plus the TRUE camera models used (for tests)."""
        dets: list[Detection] = []
        cams = self.cameras(platforms)
        for sid, cam in cams.items():
            for tgt in targets:
                d = self._camera_detection(
                    sid,
                    t,
                    tgt.state[0:3],
                    tgt.state[6:9],
                    tgt.yaw,
                    cam,
                    "vehicle",
                    0.9,
                    {"color": tgt.color},
                )
                if d is not None:
                    dets.append(d)
            for sign in self.scene.signs:
                d = self._camera_detection(
                    sid,
                    t,
                    np.asarray(sign.position),
                    sign.extent,
                    sign.yaw,
                    cam,
                    "sign",
                    0.95,
                    {"sign_type": sign.sign_type},
                )
                if d is not None:
                    dets.append(d)
            for veh in self.scene.vehicles:
                d = self._camera_detection(
                    sid,
                    t,
                    np.asarray(veh.position),
                    veh.extent,
                    veh.yaw,
                    cam,
                    "vehicle",
                    0.9,
                    {"color": veh.color},
                )
                if d is not None:
                    dets.append(d)
        for sid, s in self.cfg.sensors.items():
            if isinstance(s, RadarConfig):
                pos = np.asarray(s.position if s.position is not None else [0.0, 0.0, 0.0])
                for tgt in targets:
                    if self.rng.uniform() > self.p_detect:
                        continue
                    z = h_radar(tgt.state, pos)
                    dets.append(
                        Detection(
                            sensor_id=sid,
                            timestamp=t,
                            measurement=z + self.rng.multivariate_normal(np.zeros(4), self._R[sid]),
                            class_label="vehicle",
                            class_confidence=0.6,
                        )
                    )
        return dets, cams
