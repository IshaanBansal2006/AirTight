"""Simulated IMU: conventions, and that strapdown integration of
the synthesised readings reproduces the trajectory they came from."""

from __future__ import annotations

import numpy as np

from airtight.swarm.autonomy.executor import SmoothBackend
from airtight.swarm.autonomy.world_state import DroneState, WorldState
from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import ImuConfig
from airtight.swarm.edge.imu import ImuSynthesizer, integrate

G = np.array([0.0, 0.0, -9.81])


def noiseless(rate: float = 100.0) -> ImuConfig:
    return ImuConfig(
        rate_hz=rate,
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        accel_bias_random_walk=0.0,
        gyro_bias_random_walk=0.0,
        accel_bias_init_std=0.0,
        gyro_bias_init_std=0.0,
    )


def test_at_rest_reads_gravity_up_along_body_z() -> None:
    imu = ImuSynthesizer(noiseless(), np.random.default_rng(0))
    q = rotation.from_yaw(0.7)
    assert imu.sample(0.0, np.zeros(3), q) is None
    s = imu.sample(0.01, np.zeros(3), q)
    assert s is not None
    np.testing.assert_allclose(s.accel, [0.0, 0.0, 9.81], atol=1e-12)
    np.testing.assert_allclose(s.gyro, 0.0, atol=1e-12)


def test_forward_acceleration_and_yaw_rate_appear_in_the_body_frame() -> None:
    imu = ImuSynthesizer(noiseless(), np.random.default_rng(0))
    yaw0, rate, a = np.pi / 2, 0.3, 1.5  # heading +y, turning, accelerating along heading
    v0 = np.zeros(3)
    imu.sample(0.0, v0, rotation.from_yaw(yaw0))
    dt = 0.01
    v1 = v0 + a * dt * np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
    s = imu.sample(dt, v1, rotation.from_yaw(yaw0 + rate * dt))
    assert s is not None
    np.testing.assert_allclose(s.accel, [a, 0.0, 9.81], atol=1e-9)  # body x = heading
    np.testing.assert_allclose(s.gyro, [0.0, 0.0, rate], atol=1e-9)


def test_bias_and_noise_statistics_match_config() -> None:
    cfg = ImuConfig(rate_hz=100.0)
    imu = ImuSynthesizer(cfg, np.random.default_rng(3))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    imu.sample(0.0, np.zeros(3), q)
    reads = np.array([imu.sample(0.01 * k, np.zeros(3), q).accel for k in range(1, 4001)])
    resid = reads - [0.0, 0.0, 9.81]
    n = cfg.per_sample()
    # residual = turn-on bias + slow walk + white noise; std over 40 s is dominated by white noise
    assert abs(resid.std(axis=0).mean() - n["accel_noise"]) < 0.3 * n["accel_noise"]
    assert np.linalg.norm(resid.mean(axis=0)) < 4 * cfg.accel_bias_init_std


def test_strapdown_integration_recovers_a_smooth_flight() -> None:
    """The round trip that pins the conventions: fly a SmoothBackend patrol,
    synthesise a perfect IMU from it, integrate the readings, get the flight back."""
    w = WorldState()
    d = DroneState(drone_id="d0", position=[0.0, 0.0, 4.0], speed=3.0)
    w.update_drone(d)
    backend = SmoothBackend(w, max_accel=2.0, max_yaw_rate=1.5)
    legs = [np.array([20.0, 0.0, 4.0]), np.array([20.0, 15.0, 5.0]), np.array([0.0, 15.0, 4.0])]
    imu = ImuSynthesizer(noiseless(), np.random.default_rng(0))
    dt, t = 0.01, 0.0
    p, v, q = np.array(d.position), np.array(d.velocity), np.array(d.orientation)
    imu.sample(t, v, q)
    leg = 0
    backend.goto("d0", legs[leg])
    for _ in range(6000):
        backend.step(dt)
        t += dt
        s = imu.sample(t, np.array(d.velocity), np.array(d.orientation))
        assert s is not None
        p, v, q = integrate(p, v, q, s.accel, s.gyro, dt, G)
        if np.linalg.norm(np.array(d.position) - legs[leg]) < 0.05 and leg < len(legs) - 1:
            leg += 1
            backend.goto("d0", legs[leg])
    # dead reckoning drift here is pure discretisation error (first-order
    # integration of finite-difference accelerations) — small over a 60 s flight
    assert np.linalg.norm(p - np.array(d.position)) < 0.5
    assert np.linalg.norm(v - np.array(d.velocity)) < 0.1
    assert np.linalg.norm(rotation.boxminus(q, np.array(d.orientation))) < 0.01
    assert leg == len(legs) - 1  # the flight actually happened
