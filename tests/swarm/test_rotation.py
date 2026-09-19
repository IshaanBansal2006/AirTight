"""Quaternion / rotation-vector conventions the ego filter rests on."""

from __future__ import annotations

import numpy as np

from airtight.swarm.edge import rotation as rot


def random_quats(n: int, seed: int = 0) -> np.ndarray:
    return rot.normalize(np.random.default_rng(seed).normal(size=(n, 4)))


def test_exp_log_roundtrip_including_small_angles() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 3))
    v *= rng.uniform(0.0, 3.0, size=(200, 1)) / np.linalg.norm(v, axis=1, keepdims=True)
    v[:5] *= 1e-9  # exercise the small-angle series
    np.testing.assert_allclose(rot.log(rot.exp(v)), v, atol=1e-9)


def test_log_is_the_short_rotation_and_sign_invariant() -> None:
    q = random_quats(50)
    np.testing.assert_allclose(rot.log(q), rot.log(-q), atol=1e-12)
    assert np.all(np.linalg.norm(rot.log(q), axis=1) <= np.pi + 1e-9)


def test_matrix_is_a_rotation_and_agrees_with_rotate() -> None:
    q = random_quats(20, seed=2)
    R = rot.to_matrix(q)
    np.testing.assert_allclose(
        np.einsum("nij,nkj->nik", R, R), np.broadcast_to(np.eye(3), R.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
    v = np.array([0.3, -1.2, 2.0])
    np.testing.assert_allclose(rot.rotate(q, v), np.einsum("nij,j->ni", R, v), atol=1e-12)


def test_from_matrix_inverts_to_matrix() -> None:
    for q in random_quats(30, seed=3):
        q_back = rot.from_matrix(rot.to_matrix(q))
        np.testing.assert_allclose(rot.to_matrix(q_back), rot.to_matrix(q), atol=1e-12)


def test_boxminus_inverts_boxplus() -> None:
    q = random_quats(40, seed=4)
    d = np.random.default_rng(5).normal(scale=0.4, size=(40, 3))
    np.testing.assert_allclose(rot.boxminus(rot.boxplus(q, d), q), d, atol=1e-10)


def test_perturbation_is_in_the_body_frame() -> None:
    """q [+] d rotates first by d in the BODY frame, then by q — the right-
    perturbation convention 041 fixes. Check on a case where left and right differ."""
    q = rot.from_yaw(np.pi / 2)  # body x -> world y
    d = np.array([0.3, 0.0, 0.0])  # roll about body x
    expected = rot.multiply(q, rot.exp(d))
    np.testing.assert_allclose(rot.boxplus(q, d), expected, atol=1e-12)
    assert not np.allclose(rot.boxplus(q, d), rot.multiply(rot.exp(d), q))


def test_yaw_roundtrip_and_wrap() -> None:
    for yaw in [-3.0, -1.0, 0.0, 0.7, 2.9]:
        assert abs(rot.yaw_of(rot.from_yaw(yaw)) - yaw) < 1e-12
    assert abs(rot.wrap_angle(np.pi + 0.1) - (-np.pi + 0.1)) < 1e-12
    assert abs(rot.wrap_angle(-np.pi - 0.1) - (np.pi - 0.1)) < 1e-12


def test_batched_shapes() -> None:
    q = random_quats(7)
    assert rot.multiply(q, q).shape == (7, 4)
    assert rot.to_matrix(q).shape == (7, 3, 3)
    assert rot.exp(np.zeros((7, 3))).shape == (7, 4)
    assert rot.rotate(q, np.ones((7, 3))).shape == (7, 3)
    assert rot.rotate(q[0], np.ones(3)).shape == (3,)
