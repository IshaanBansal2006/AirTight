"""Simulated IMU: the ego filter's odometry source.

Turns a platform's TRUE trajectory into what a strapdown IMU on it would
report — specific force and angular rate in the body frame, corrupted by a
slowly wandering bias and white noise — plus the strapdown integration that
turns those reports back into motion (used by the filter's predict step, and
by the tests to prove the two agree).

Conventions (rotation.py): q maps body -> world; angular rate
is a body-frame vector; gravity is a world-frame vector pointing down, so a
platform at rest reads specific force +9.81 along body z (up).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge import rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from airtight.swarm.edge.config import ImuConfig


@dataclass
class ImuSample:
    timestamp: float
    accel: NDArray[np.float64]  # specific force, body frame, m/s^2
    gyro: NDArray[np.float64]  # angular rate, body frame, rad/s


@dataclass
class ImuSynthesizer:
    """One per platform. Feed it the true pose/velocity at the IMU rate.

    Acceleration and angular rate come from finite differences of consecutive
    truth samples — a synthesised IMU has no dynamics of its own to read, so
    the trajectory must be smooth enough to difference (SmoothBackend, 019).
    Biases start from a random turn-on value and random-walk from there; the
    true biases are exposed so a test can check the filter estimates them.
    """

    config: ImuConfig
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.accel_bias = self.rng.normal(0.0, self.config.accel_bias_init_std, 3)
        self.gyro_bias = self.rng.normal(0.0, self.config.gyro_bias_init_std, 3)
        self._prev: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None

    def sample(
        self,
        t: float,
        velocity: NDArray[np.float64],
        orientation: NDArray[np.float64],
    ) -> ImuSample | None:
        """The IMU reading for the interval ending at `t`; None on the first call."""
        v = np.asarray(velocity, dtype=float)
        q = rotation.normalize(np.asarray(orientation, dtype=float))
        if self._prev is None:
            self._prev = (t, v, q)
            return None
        t0, v0, q0 = self._prev
        dt = t - t0
        if dt <= 0.0:
            raise ValueError(f"IMU samples must advance in time: got t={t} after t={t0}")
        self._prev = (t, v, q)

        accel_w = (v - v0) / dt
        # specific force is what the accelerometer feels: acceleration minus
        # gravity. Expressed in the body frame at the START of the interval —
        # the frame `integrate` rotates it back out with — so a noiseless
        # reading integrates to exactly the trajectory it came from.
        f_b = rotation.rotate(rotation.conjugate(q0), accel_w - np.asarray(self.config.gravity))
        omega_b = rotation.boxminus(q, q0) / dt  # body-frame rotation over the interval

        n = self.config.per_sample()
        self.accel_bias = self.accel_bias + self.rng.normal(0.0, n["accel_bias_walk"], 3)
        self.gyro_bias = self.gyro_bias + self.rng.normal(0.0, n["gyro_bias_walk"], 3)
        return ImuSample(
            timestamp=t,
            accel=f_b + self.accel_bias + self.rng.normal(0.0, n["accel_noise"], 3),
            gyro=omega_b + self.gyro_bias + self.rng.normal(0.0, n["gyro_noise"], 3),
        )


def integrate(
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    orientation: NDArray[np.float64],
    accel: NDArray[np.float64],
    gyro: NDArray[np.float64],
    dt: float,
    gravity: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """One strapdown step (batched over a leading sigma-point dimension).

    Rotate the bias-corrected specific force into the world frame with the
    orientation at the START of the interval, add gravity back, integrate.
    Attitude advances by the body rotation vector gyro*dt on the right — the
    same retraction the error state uses, so predict and error are consistent.
    """
    a_w = rotation.rotate(orientation, accel) + gravity
    p_new = position + velocity * dt + 0.5 * a_w * dt * dt
    v_new = velocity + a_w * dt
    q_new = rotation.boxplus(orientation, gyro * dt)
    return p_new, v_new, q_new
