"""Offline regression tests for the 9-D UKF + observation models.

No ROS required — synthesizes camera+radar measurements from a constant-velocity
ground truth through the real h functions (same harness pattern as the thin-slice
node) and asserts the filter converges and stays numerically healthy. Guards the
live-debugged failure modes of 2026-07-29 (see explanations/edge/ukf.md war story).
"""

from __future__ import annotations

import numpy as np
import pytest

from airtight.swarm.edge.config import UKFConfig
from airtight.swarm.edge.observation import CameraModel, h_camera, h_radar
from airtight.swarm.edge.types import TrackState
from airtight.swarm.edge.ukf import UKF

TRUE_EXTENT = np.array([0.2, 0.2, 0.2])
VEL = np.array([0.5, 0.2, 0.0])
DT = 0.1
R_CAMERA = np.diag([2.0**2, 2.0**2, 3.0**2, 3.0**2])
R_RADAR = np.diag([0.1**2, 0.01**2, 0.01**2, 0.1**2])  # 3-D radar: range, az, el, doppler
RADAR_POS = np.zeros(3)


def make_ukf() -> UKF:
    # alpha=1, kappa=1 -> all sigma weights positive at n=9 (war-story fix #1)
    return UKF(
        UKFConfig(
            state_dim=9,
            alpha=1.0,
            beta=2.0,
            kappa=1.0,
            dt=DT,
            process_noise=[1e-3] * 3 + [1e-2] * 3 + [1e-3] * 3,  # extent Q per D-B11
        )
    )


def chase_camera(target_pos: np.ndarray) -> CameraModel:
    cam_pos = target_pos + np.array([-6.0, -2.0, 3.0])
    z_c = (target_pos - cam_pos) / np.linalg.norm(target_pos - cam_pos)
    x_c = np.cross(z_c, [0.0, 0.0, 1.0])
    x_c = x_c / np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    return CameraModel(
        fx=480.0,
        fy=480.0,
        cx=480.0,
        cy=300.0,
        width=960,
        height=600,
        R_wc=np.stack([x_c, y_c, z_c]),
        t_w=cam_pos,
    )


def test_factory_default_is_srukf() -> None:
    """covariance_form defaults to srukf; factory dispatches."""
    from airtight.swarm.edge.filters import initial_state, make_filter
    from airtight.swarm.edge.srukf import SquareRootUKF, SRTrackState

    filt = make_filter(UKFConfig())
    assert isinstance(filt, SquareRootUKF)
    st = initial_state(filt, np.zeros(9), np.eye(9))
    assert isinstance(st, SRTrackState)
    np.testing.assert_allclose(st.S @ st.S.T, np.eye(9), atol=1e-12)


def test_sigma_weights_positive_and_normalized() -> None:
    ukf = make_ukf()
    assert np.isclose(ukf.Wm.sum(), 1.0)
    # the entire point of alpha=1, kappa=1 at n=9: no negative weights
    assert (ukf.Wm > 0).all()
    assert (ukf.Wc > 0).all()


def test_sigma_points_reproduce_covariance() -> None:
    """Regression for the rows-vs-columns Cholesky bug (found 2026-07-29).

    The weighted deviations of the sigma set must reconstruct P exactly. Using
    ROWS of numpy's lower Cholesky factor reproduces L^T L instead of L L^T=P —
    right scale, wrong correlation directions. Uses a strongly correlated P so
    the two matrices differ clearly.
    """
    ukf = make_ukf()
    rng = np.random.default_rng(11)
    A = rng.normal(size=(9, 9))
    P = A @ A.T + 0.5 * np.eye(9)  # random SPD with strong off-diagonals
    state = TrackState(x=np.zeros(9), P=P)
    sigma = ukf.generate_sigma_points(state)
    recon = np.zeros((9, 9))
    for i in range(sigma.shape[0]):
        d = sigma[i] - np.zeros(9)
        recon += ukf.Wc[i] * np.outer(d, d)
    # Wc reconstruction includes the beta correction on the center point; with
    # x0 = mean the center deviation is zero, so recon must equal P exactly.
    np.testing.assert_allclose(recon, P, atol=1e-8)


def test_srukf_matches_ukf_on_linear_case() -> None:
    """SR-UKF and UKF must agree (same math, different covariance carrier)."""
    from airtight.swarm.edge.srukf import SquareRootUKF, SRTrackState

    ukf = make_ukf()
    sr = SquareRootUKF(ukf.cfg)
    x0 = np.concatenate([np.array([1.0, -2.0, 0.5]), VEL, TRUE_EXTENT])
    P0 = np.diag([0.3] * 3 + [0.2] * 3 + [0.1] * 3)
    a = ukf.predict(TrackState(x=x0.copy(), P=P0.copy()))
    b = sr.predict(SRTrackState(x=x0.copy(), S=np.linalg.cholesky(P0)))
    np.testing.assert_allclose(a.x, b.x, atol=1e-9)
    np.testing.assert_allclose(a.P, b.S @ b.S.T, atol=1e-8)

    # one radar update on identical states must agree too
    rng_m = np.random.default_rng(5)
    z = h_radar(x0, RADAR_POS) + rng_m.multivariate_normal(np.zeros(4), R_RADAR)
    a2, ia, Sa = ukf.update(a, z, lambda x: h_radar(x, RADAR_POS), R_RADAR)
    b2, ib, Sb = sr.update(b, z, lambda x: h_radar(x, RADAR_POS), R_RADAR)
    np.testing.assert_allclose(a2.x, b2.x, atol=1e-6)
    np.testing.assert_allclose(ia, ib, atol=1e-9)
    np.testing.assert_allclose(Sa, Sb, atol=1e-6)
    np.testing.assert_allclose(a2.P, b2.S @ b2.S.T, atol=1e-6)


def test_predict_propagates_constant_velocity() -> None:
    ukf = make_ukf()
    x = np.concatenate([np.array([1.0, 2.0, 0.5]), VEL, TRUE_EXTENT])
    state = TrackState(x=x.copy(), P=np.eye(9) * 1e-4)
    pred = ukf.predict(state)
    np.testing.assert_allclose(pred.x[0:3], x[0:3] + VEL * DT, atol=1e-6)
    np.testing.assert_allclose(pred.x[3:6], VEL, atol=1e-6)  # velocity unchanged
    np.testing.assert_allclose(pred.P, pred.P.T)  # symmetric


def test_full_fusion_converges_and_stays_psd() -> None:
    rng = np.random.default_rng(7)
    ukf = make_ukf()
    pos = np.array([0.0, 0.0, 0.5])

    # two-point-differencing style init (war-story fix #2: never zero-velocity)
    x0 = np.concatenate(
        [pos + rng.normal(0, 0.5, 3), VEL + rng.normal(0, 0.3, 3), np.array([0.5, 0.5, 0.5])]
    )
    track = TrackState(x=x0, P=np.diag([0.25] * 9))
    extent_err0 = float(np.linalg.norm(x0[6:9] - TRUE_EXTENT))

    for k in range(1, 301):
        pos = np.array([0.0, 0.0, 0.5]) + VEL * (k * DT)
        x_true = np.concatenate([pos, VEL, TRUE_EXTENT])
        cam = chase_camera(pos)
        z_cam = h_camera(x_true, cam) + rng.multivariate_normal(np.zeros(4), R_CAMERA)
        z_rad = h_radar(x_true, RADAR_POS) + rng.multivariate_normal(np.zeros(4), R_RADAR)

        track = ukf.predict(track)
        # radar before camera (war-story fix #3: mild nonlinearity first)
        track, _, S_rad = ukf.update(track, z_rad, lambda x: h_radar(x, RADAR_POS), R_RADAR)
        track, _, _ = ukf.update(track, z_cam, lambda x, cam=cam: h_camera(x, cam), R_CAMERA)

        # numerical health every step: symmetric, eigenvalues non-negative (tolerance)
        np.testing.assert_allclose(track.P, track.P.T, atol=1e-9)
        assert np.linalg.eigvalsh(track.P).min() > -1e-9

    pos_err = float(np.linalg.norm(track.x[0:3] - pos))
    vel_err = float(np.linalg.norm(track.x[3:6] - VEL))
    extent_err = float(np.linalg.norm(track.x[6:9] - TRUE_EXTENT))

    assert pos_err < 0.3, f"position did not converge: {pos_err:.3f} m"
    assert vel_err < 0.3, f"velocity did not converge: {vel_err:.3f} m/s"
    # Extent is weakly observable AND structurally biased from a fixed viewpoint
    # (D-B11): with the honesty-restoring Q_ext the estimate stays plastic, so
    # require progress from the prior, not arrival at truth.
    assert extent_err < 0.9 * extent_err0, (
        f"extent not converging: {extent_err:.3f} vs initial {extent_err0:.3f}"
    )
    # innovation covariance from the last radar update is usable for gating (JPDA hook)
    assert S_rad.shape == (4, 4)


def test_update_rejects_nothing_but_stays_finite_far_from_truth() -> None:
    """A grossly wrong prior must not produce NaNs/crashes (robustness floor)."""
    rng = np.random.default_rng(3)
    ukf = make_ukf()
    pos = np.array([5.0, -3.0, 0.5])
    x_true = np.concatenate([pos, VEL, TRUE_EXTENT])
    cam = chase_camera(pos)
    z_cam = h_camera(x_true, cam) + rng.multivariate_normal(np.zeros(4), R_CAMERA)

    bad = TrackState(
        x=np.concatenate([pos + 5.0, VEL + 1.0, np.array([1.0, 1.0, 1.0])]),
        P=np.diag([4.0] * 9),
    )
    bad = ukf.predict(bad)
    bad, innovation, S = ukf.update(bad, z_cam, lambda x, cam=cam: h_camera(x, cam), R_CAMERA)
    assert np.isfinite(bad.x).all()
    assert np.isfinite(bad.P).all()
    assert np.isfinite(innovation).all() and np.isfinite(S).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
