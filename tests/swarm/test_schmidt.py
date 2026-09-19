"""Schmidt-Kalman consider update: reduces to the plain update
when the pose is exact, widens the innovation when it is not, and — the reason
it exists — stays honest when every look comes through the same biased pose."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import CameraConfig, TrackerConfig, UKFConfig
from airtight.swarm.edge.filters import initial_state, make_filter
from airtight.swarm.edge.observation import h_camera
from airtight.swarm.edge.schmidt import ConsiderCamera, ConsiderPose, MountedCamera, cv_transition
from airtight.swarm.edge.sensing import camera_model
from airtight.swarm.edge.tracker import MultiTargetTracker
from airtight.swarm.edge.types import Detection
from airtight.swarm.scene import demo_scene
from tests.swarm.road_harness import RoadFlight, road_waypoints

CAM = CameraConfig(platform="d0")
R = np.diag(np.square(np.asarray(CAM.measurement_noise)))
# moving along +x: a STATIC target with velocity uncertainty would put the
# velocity-derived yaw (013 option A) at its undefined point, and the projected
# box would swing with every velocity sigma point — a model limit, not a
# consider-filter property
X_TRUE = np.array([20.0, 0.5, 0.75, 2.0, 0.0, 0.0, 4.5, 1.8, 1.5])
DRONE_P, DRONE_YAW = np.array([0.0, 0.0, 4.0]), 0.0


def mounted(pos: np.ndarray, yaw: float, pos_std: float, ang_std: float) -> MountedCamera:
    cov = np.diag([pos_std**2] * 3 + [ang_std**2] * 3)
    return MountedCamera(CAM, ConsiderPose("d0", pos, rotation.from_yaw(yaw), cov))


def test_zero_pose_covariance_reduces_to_the_plain_update() -> None:
    cfg = UKFConfig()
    filt = make_filter(cfg)
    # near-linear regime: the 15-point and 9-point unscented transforms are two
    # second-order approximations of the same nonlinear h and only coincide as
    # the spread -> 0 (at 0.1 m std they differ by ~3%; at 0.01 m, by ~0.1%)
    P0 = np.diag([1e-4] * 9)
    state = initial_state(filt, X_TRUE.copy(), P0)
    mc = mounted(DRONE_P, DRONE_YAW, 0.0, 0.0)
    z = h_camera(X_TRUE, mc.camera()) + np.array([1.0, -1.0, 0.5, 0.5])
    cc = ConsiderCamera(mc, R, cfg)
    s_c, P_xc, nu_c, S_c = cc.update(state, np.zeros((9, 6)), z)
    s_p, nu_p, S_p = filt.update(state, z, lambda x: h_camera(x, mc.camera()), R)
    np.testing.assert_allclose(S_c, S_p, rtol=2e-3, atol=1e-2)  # px^2; cross terms ~1e-3
    np.testing.assert_allclose(s_c.x, s_p.x, atol=1e-4)
    np.testing.assert_allclose(s_c.S @ s_c.S.T, s_p.S @ s_p.S.T, rtol=5e-3, atol=1e-7)  # diag ~1e-4
    np.testing.assert_allclose(P_xc, 0.0, atol=1e-9)


def test_pose_uncertainty_widens_the_innovation_covariance() -> None:
    cfg = UKFConfig()
    filt = make_filter(cfg)
    state = initial_state(filt, X_TRUE.copy(), np.diag([0.25] * 3 + [0.1] * 3 + [0.05] * 3))
    exact = ConsiderCamera(mounted(DRONE_P, DRONE_YAW, 0.0, 0.0), R, cfg)
    uncertain = ConsiderCamera(mounted(DRONE_P, DRONE_YAW, 0.5, np.deg2rad(2.0)), R, cfg)
    _, S_e = exact.measurement_prediction(state, np.zeros((9, 6)))
    _, S_u = uncertain.measurement_prediction(state, np.zeros((9, 6)))
    assert np.all(np.linalg.eigvalsh(S_u - S_e) > -1e-9)  # S_u >= S_e
    assert np.trace(S_u) > 2.0 * np.trace(S_e)


def test_consider_stays_consistent_under_a_biased_pose_and_the_plain_split_does_not() -> None:
    """Sixty looks at a constant-velocity target through a camera whose ESTIMATED
    pose is 0.4 m / 1.5 deg off (a ~1-sigma draw from its own covariance). The
    plain split converges confidently to the wrong place; the consider filter's
    covariance floors at the pose error it was told about."""
    rng = np.random.default_rng(7)
    cfg = UKFConfig(process_noise=[1e-6] * 9)
    filt = make_filter(cfg)
    P0 = np.diag([1.0] * 3 + [0.01] * 3 + [0.09] * 3)
    pos_std, ang_std = 0.4, np.deg2rad(1.5)
    true_pose = (DRONE_P, DRONE_YAW)
    est_pose = (DRONE_P + np.array([0.0, pos_std, 0.0]), DRONE_YAW + ang_std)
    truth_cam = camera_model(true_pose[0], rotation.from_yaw(true_pose[1]), CAM)
    mc = mounted(est_pose[0], est_pose[1], pos_std, ang_std)
    consider = ConsiderCamera(mc, R, cfg)
    F = cv_transition(cfg.dt)

    s_plain = initial_state(filt, X_TRUE.copy(), P0)
    s_cons = initial_state(filt, X_TRUE.copy(), P0)
    P_xc = np.zeros((9, 6))
    x_true = X_TRUE.copy()
    for _ in range(60):
        x_true[0:3] += x_true[3:6] * cfg.dt
        z = h_camera(x_true, truth_cam) + rng.multivariate_normal(np.zeros(4), R)
        s_plain = filt.predict(s_plain)
        s_plain, _, _ = filt.update(s_plain, z, lambda x: h_camera(x, mc.camera()), R)
        s_cons = filt.predict(s_cons)
        P_xc = F @ P_xc
        s_cons, P_xc, _, _ = consider.update(s_cons, P_xc, z)

    def nees_pos(s) -> float:
        e = s.x[0:3] - x_true[0:3]
        P = (s.S @ s.S.T)[0:3, 0:3]
        return float(e @ np.linalg.solve(P, e))

    n_plain, n_cons = nees_pos(s_plain), nees_pos(s_cons)
    assert n_plain > 30.0, (n_plain, n_cons)  # ideal 3: the plain split is wildly overconfident
    assert n_cons < 12.0, (n_plain, n_cons)
    assert n_plain > 5.0 * n_cons
    # the cross-covariance is doing the remembering: nonzero after correlated looks
    assert np.abs(P_xc).max() > 1e-4
    assert consider.repairs == 0


def test_tracker_accepts_a_mounted_camera_and_carries_the_consider_block() -> None:
    # target starts 18 m ahead so the forward camera sees it (see test_ego)
    flight = RoadFlight(seed=5, scene=replace(demo_scene(), target_start=(10.0, 0.0, 0.75)))
    cfg: TrackerConfig = flight.cfg
    tracker = MultiTargetTracker(cfg)
    confirmed = []
    for frame in flight.frames({"d0": road_waypoints()}, duration_s=8.0):
        d = frame.truth["d0"]
        mc = MountedCamera(
            cfg.sensors["cam_d0"],
            ConsiderPose(
                "d0",
                np.asarray(d.position),
                np.asarray(d.orientation),
                np.diag([0.05**2] * 3 + [np.deg2rad(0.5) ** 2] * 3),
            ),
        )
        dets = [
            x
            for x in frame.detections
            if x.sensor_id == "radar_gs"
            or (x.sensor_id == "cam_d0" and x.attributes.get("color") == "silver")
        ]
        confirmed = tracker.step(dets, camera_models={"cam_d0": mc})
    assert len(confirmed) == 1
    tr = confirmed[0]
    err = np.linalg.norm(tr.state.x[0:3] - frame.target.state[0:3])
    assert err < 1.0, err
    assert "d0" in tr.consider_xc and np.abs(tr.consider_xc["d0"]).max() > 0.0
    assert tracker.consider_repairs == 0


def test_legacy_bare_camera_model_path_is_unchanged() -> None:
    """A CameraModel value still means 'pose exact' — the existing tests' harness."""
    cfg = TrackerConfig()
    tracker = MultiTargetTracker(cfg)
    cam = camera_model(DRONE_P, rotation.from_yaw(DRONE_YAW), CAM)
    det = Detection(sensor_id="cam_front", timestamp=0.1, measurement=h_camera(X_TRUE, cam))
    tracker.step([det], camera_models={"cam_front": cam})
    assert tracker.tracks == []  # camera alone never births (unchanged behaviour)
