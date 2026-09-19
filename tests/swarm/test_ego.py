"""Ego-pose SLAM filter (decisions 018/019): dead-reckoning honesty, sign fixes,
delayed landmark initialisation, and moving-object rejection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import EgoConfig
from airtight.swarm.edge.ego import EgoFilter
from airtight.swarm.scene import demo_scene
from tests.swarm.road_harness import RoadFlight, road_waypoints


def make_filter(flight: RoadFlight, did: str = "d0", cfg: EgoConfig | None = None) -> EgoFilter:
    d = flight.world.drones[did]
    return EgoFilter(
        cfg or EgoConfig(imu=flight.imu_cfg),
        flight.scene.signs,
        np.asarray(d.position),
        np.asarray(d.velocity),
        np.asarray(d.orientation),
    )


def pose_error(f: EgoFilter, truth) -> tuple[float, float]:
    dp = float(np.linalg.norm(f.p - np.asarray(truth.position)))
    dth = float(np.linalg.norm(rotation.boxminus(f.q, np.asarray(truth.orientation))))
    return dp, dth


def nees(f: EgoFilter, truth) -> float:
    e = np.concatenate(
        [f.p - np.asarray(truth.position), rotation.boxminus(f.q, np.asarray(truth.orientation))]
    )
    return float(e @ np.linalg.solve(f.pose_cov(), e))


def test_dead_reckoning_drifts_and_the_covariance_says_so() -> None:
    flight = RoadFlight(seed=1)
    f = make_filter(flight)
    errs, ratios = [], []
    for frame in flight.frames({"d0": road_waypoints()}, duration_s=20.0):
        for s in frame.imu["d0"]:
            f.predict(s)
        dp, _ = pose_error(f, frame.truth["d0"])
        sigma = float(np.sqrt(np.trace(f.P[0:3, 0:3])))
        errs.append(dp)
        ratios.append(dp / sigma)
    assert errs[-1] > errs[20]  # drift grows
    assert errs[-1] > 0.5  # IMU-only over 20 s is metres, not centimetres
    assert np.mean(np.asarray(ratios) < 3.0) > 0.9  # ...and the filter knows it


def test_sign_fixes_bound_the_error_and_stay_consistent() -> None:
    flight = RoadFlight(seed=2)
    f = make_filter(flight)
    cam_cfg = flight.cfg.sensors["cam_d0"]
    R = np.diag(np.square(np.asarray(cam_cfg.measurement_noise)))
    errs, nees_vals, signs_used = [], [], 0
    for frame in flight.frames({"d0": road_waypoints()}, duration_s=22.0):
        for s in frame.imu["d0"]:
            f.predict(s)
        dets = [d for d in frame.detections if d.sensor_id == "cam_d0" and d.class_label == "sign"]
        rep = f.update(dets, cam_cfg, frame.t, R)
        signs_used += rep.signs_used
        dp, dth = pose_error(f, frame.truth["d0"])
        errs.append(dp)
        nees_vals.append(nees(f, frame.truth["d0"]))
    assert signs_used > 50
    assert np.mean(errs) < 0.4 and errs[-1] < 0.6
    assert dth < np.deg2rad(2.0)
    # 6-DOF NEES: ideal 6; a filter that is badly overconfident sits in the hundreds
    assert np.median(nees_vals) < 20.0
    assert f.repairs == 0


def test_parked_vehicle_becomes_a_landmark_but_the_moving_one_does_not() -> None:
    # start the target 18 m ahead so the forward camera actually sees it while
    # the drone closes on it (from the default 3 m ahead it is overtaken unseen)
    scene = replace(demo_scene(), target_start=(10.0, 0.0, 0.75))
    flight = RoadFlight(seed=3, scene=scene)
    f = make_filter(flight)
    cam_cfg = flight.cfg.sensors["cam_d0"]
    R = np.diag(np.square(np.asarray(cam_cfg.measurement_noise)))
    rejected = 0
    for frame in flight.frames({"d0": road_waypoints()}, duration_s=22.0):
        for s in frame.imu["d0"]:
            f.predict(s)
        dets = [d for d in frame.detections if d.sensor_id == "cam_d0"]  # signs AND all vehicles
        rep = f.update(dets, cam_cfg, frame.t, R)
        rejected += rep.candidates_rejected
    colours = {lm.color for lm in f.landmarks}
    assert "silver" not in colours  # the moving target never passed the static check
    assert rejected > 0
    assert len(f.landmarks) >= 2
    for lm in f.landmarks:
        truth = next(v for v in flight.scene.vehicles if v.color == lm.color)
        assert np.linalg.norm(lm.position - np.asarray(truth.position)) < 1.5, lm.color
        assert lm.hits >= 3


def test_pose_message_carries_a_valid_factor() -> None:
    flight = RoadFlight(seed=4)
    f = make_filter(flight)
    msg = f.to_msg("d0", 0.0)
    L = np.asarray(msg.pose_sqrt_cov).reshape(6, 6)
    assert np.allclose(np.triu(L, 1), 0.0)
    np.testing.assert_allclose(L @ L.T, f.pose_cov(), atol=1e-12)
    assert abs(np.linalg.norm(msg.orientation) - 1.0) < 1e-12
