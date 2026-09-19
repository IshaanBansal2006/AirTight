"""Integration test: multi-sensor tracker lifecycle on synthetic detections.

Exercises birth (two-point radar initiation) -> confirmation -> tracking
accuracy with camera+radar PDA fusion -> deletion on missed detections.
"""

from __future__ import annotations

import numpy as np

from airtight.swarm.edge.config import TrackerConfig
from airtight.swarm.edge.observation import CameraModel, h_camera, h_radar
from airtight.swarm.edge.tracker import MultiTargetTracker
from airtight.swarm.edge.types import Detection

TRUE_EXTENT = np.array([0.2, 0.2, 0.2])
VEL = np.array([0.5, 0.2, 0.0])
RADAR_POS = np.zeros(3)
R_CAM = np.diag([2.0**2, 2.0**2, 3.0**2, 3.0**2])
R_RAD = np.diag([0.1**2, 0.01**2, 0.01**2, 0.1**2])


def chase_camera(target_pos: np.ndarray) -> CameraModel:
    cam_pos = target_pos + np.array([-6.0, -2.0, 3.0])
    z_c = (target_pos - cam_pos) / np.linalg.norm(target_pos - cam_pos)
    x_c = np.cross(z_c, [0.0, 0.0, 1.0])
    x_c /= np.linalg.norm(x_c)
    return CameraModel(
        fx=480.0,
        fy=480.0,
        cx=480.0,
        cy=300.0,
        width=960,
        height=600,
        R_wc=np.stack([x_c, np.cross(z_c, x_c), z_c]),
        t_w=cam_pos,
    )


def test_tracker_full_lifecycle() -> None:
    rng = np.random.default_rng(4)
    cfg = TrackerConfig()  # defaults: cam_front + radar_gs at origin, SR-UKF
    tracker = MultiTargetTracker(cfg)
    dt = cfg.ukf.dt
    pos0 = np.array([2.0, 1.0, 0.5])

    confirmed_at = None
    for k in range(1, 81):
        pos = pos0 + VEL * (k * dt)
        x_true = np.concatenate([pos, VEL, TRUE_EXTENT])
        cam = chase_camera(pos)
        dets = [
            Detection(
                sensor_id="radar_gs",
                timestamp=k * dt,
                measurement=h_radar(x_true, RADAR_POS)
                + rng.multivariate_normal(np.zeros(4), R_RAD),
                class_label="vehicle",
                class_confidence=0.5,
            ),
            Detection(
                sensor_id="cam_front",
                timestamp=k * dt,
                measurement=h_camera(x_true, cam) + rng.multivariate_normal(np.zeros(4), R_CAM),
                class_label="vehicle",
                class_confidence=0.8,
            ),
        ]
        confirmed = tracker.step(dets, camera_models={"cam_front": cam})
        if confirmed and confirmed_at is None:
            confirmed_at = k

    # birth needs 2 cycles (two-point init), confirmation confirm_hits more
    assert confirmed_at is not None, "no track was ever confirmed"
    assert confirmed_at <= 3 + cfg.confirm_hits
    assert len(confirmed) == 1, f"expected exactly 1 confirmed track, got {len(confirmed)}"

    # tracking accuracy after 80 fused cycles
    est = confirmed[0].state.x
    assert float(np.linalg.norm(est[0:3] - pos)) < 0.5
    assert float(np.linalg.norm(est[3:6] - VEL)) < 0.5

    # classification fused across sensors + time (012/015/D-B7): DS-native
    # masses on the track, pignistic decision resolves to the true class
    from airtight.swarm.edge.classification import DSClassifier

    beliefs = confirmed[0].class_beliefs
    assert beliefs and abs(sum(beliefs.values()) - 1.0) < 1e-6
    label, conf = DSClassifier().decide(beliefs)
    assert label == "vehicle" and conf > 0.8

    # deletion: stop feeding detections -> track coasts then dies
    tid = confirmed[0].track_id
    for _ in range(cfg.max_misses):
        tracker.step([], camera_models={})
    assert all(t.track_id != tid for t in tracker.tracks), "track not deleted after misses"


def test_tracker_no_false_births_from_camera() -> None:
    """Camera detections alone must not create tracks (no depth -> no birth)."""
    rng = np.random.default_rng(9)
    cfg = TrackerConfig()
    tracker = MultiTargetTracker(cfg)
    pos = np.array([2.0, 1.0, 0.5])
    for k in range(1, 10):
        x_true = np.concatenate([pos, VEL, TRUE_EXTENT])
        cam = chase_camera(pos)
        det = Detection(
            sensor_id="cam_front",
            timestamp=k * cfg.ukf.dt,
            measurement=h_camera(x_true, cam) + rng.multivariate_normal(np.zeros(4), R_CAM),
        )
        tracker.step([det], camera_models={"cam_front": cam})
    assert tracker.tracks == []
