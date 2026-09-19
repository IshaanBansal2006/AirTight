"""Quaternion and rotation-vector utilities for the ego-pose state.

Conventions, fixed here and used everywhere an ego pose appears:
  - Quaternions are Hamilton, scalar-first [w, x, y, z], unit norm.
  - q rotates BODY vectors into the WORLD frame: v_w = R(q) v_b.
  - Small rotations are rotation vectors (axis * angle, rad) in the BODY frame,
    applied on the right: q [+] dtheta = q (x) exp(dtheta). The ego filter's
    orientation error is exactly this local perturbation (Sola 2017, "Quaternion
    kinematics for the error-state Kalman filter").
  - Every function accepts a leading batch dimension so a whole sigma-point set
    goes through in one call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SMALL_ANGLE = 1e-6


def normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(q / np.linalg.norm(q, axis=-1, keepdims=True), dtype=float)


def multiply(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product a (x) b (batched over leading dims)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def conjugate(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a unit quaternion."""
    return np.asarray(q * np.array([1.0, -1.0, -1.0, -1.0]), dtype=float)


def to_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix R(q), shape (..., 3, 3): v_w = R v_b."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def from_matrix(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit quaternion of one rotation matrix (Shepperd's method: pick the
    largest diagonal-based term so no division is near zero)."""
    m = np.asarray(R, dtype=float)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    out = np.asarray(q, dtype=float)
    return out if out[0] >= 0.0 else -out


def exp(rotvec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation vector -> unit quaternion (the exponential map), batched."""
    v = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * angle
    small = angle < _SMALL_ANGLE
    safe = np.where(small, 1.0, angle)
    # sin(a/2)/a -> 1/2 as a -> 0; the series keeps the map smooth at zero
    k = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / safe)
    return np.concatenate([np.cos(half), k * v], axis=-1)


def log(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit quaternion -> rotation vector (the log map), batched.

    Canonicalises to w >= 0 first so the result is the SHORT rotation (angle in
    [0, pi]); q and -q are the same rotation and must give the same vector.
    """
    q = np.where(q[..., :1] < 0.0, -q, q)
    w = q[..., :1]
    vec = q[..., 1:]
    s = np.linalg.norm(vec, axis=-1, keepdims=True)
    small = s < _SMALL_ANGLE
    safe = np.where(small, 1.0, s)
    angle = 2.0 * np.arctan2(s, w)
    k = np.where(small, 2.0 / np.maximum(w, _SMALL_ANGLE), angle / safe)
    return k * vec


def boxplus(q: NDArray[np.float64], dtheta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Retraction: apply a body-frame rotation vector on the right."""
    return normalize(multiply(q, exp(dtheta)))


def boxminus(q_a: NDArray[np.float64], q_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Local difference: the rotation vector d with q_b [+] d == q_a."""
    return log(normalize(multiply(conjugate(q_b), q_a)))


def rotate(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """R(q) v, batched over leading dims of either argument."""
    return np.asarray(np.einsum("...ij,...j->...i", to_matrix(q), v), dtype=float)


def from_yaw(yaw: float) -> NDArray[np.float64]:
    """Level attitude with heading `yaw` (rad, about world z)."""
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def yaw_of(q: NDArray[np.float64]) -> float:
    """Heading: the world-frame direction of the body x axis."""
    fwd = rotate(np.asarray(q, dtype=float), np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(fwd[1], fwd[0]))


def wrap_angle(a: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Wrap to [-pi, pi)."""
    return np.asarray((np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi, dtype=float)
