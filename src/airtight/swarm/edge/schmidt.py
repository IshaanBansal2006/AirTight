"""Schmidt-Kalman ("consider") camera update for the target tracker.

The target tracker's camera now rides a drone whose pose is an ESTIMATE with a
covariance (ego.py). Using that pose as if it were exact makes the tracker
overconfident twice over: each update ignores the pose error, and successive
updates from the same drone treat that one error as fresh independent noise.

The consider formulation (Schmidt 1966) fixes both without estimating the pose
here. Take the joint of the target state x (9) and the pose error c (6):

    P_joint = [[P_xx, P_xc],
               [P_xc', P_cc]]        P_cc from the ego filter, never updated here

push joint sigma points through h(x, c) = h_camera(x, camera at pose [+] c),
and update ONLY x, with the joint's cross terms doing the accounting:

    K      = P_xz S^-1                      (gain for x alone; K_c = 0)
    x+     = x + K nu
    P_xx+  = P_xx - K S K'
    P_xc+  = P_xc - K P_cz'                 (the block the plain split forgets)
    P_cc+  = P_cc                           (consider: carried, not corrected)

The innovation covariance S includes the pose term through the sigma points,
so gating and PDA see the honest spread. Between updates P_xc propagates with
the target's transition (F P_xc) while the pose error is treated as constant —
the standard slowly-varying-consider approximation, stated in 019.

Works on either state carrier: the SR-UKF's factor is squared to P for this
step and re-factorised after (with the same repair path), because the
consider update is a two-block operation the rank-1 machinery doesn't cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge import rotation
from airtight.swarm.edge.observation import CameraModel, h_camera
from airtight.swarm.edge.sensing import mount_rotation
from airtight.swarm.edge.srukf import SRTrackState
from airtight.swarm.edge.types import TrackState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from airtight.swarm.edge.config import CameraConfig, UKFConfig

POSE_ERR_DIM = 6


@dataclass
class ConsiderPose:
    """An observing platform as the tracker sees it: estimated pose + the 6x6
    covariance of its error [dp, dtheta] (the ego filter's pose_cov())."""

    drone_id: str
    position: NDArray[np.float64]
    orientation: NDArray[np.float64]  # [w, x, y, z]
    cov: NDArray[np.float64]  # (6, 6)


@dataclass
class MountedCamera:
    """What the tracker gets per cycle for a drone-borne camera: the sensor's
    intrinsics + mount, and the platform's estimated pose to hang it on."""

    cfg: CameraConfig
    pose: ConsiderPose

    def camera(self, dc: NDArray[np.float64] | None = None) -> CameraModel:
        """The CameraModel at the estimated pose perturbed by a pose error dc."""
        p, q = self.pose.position, self.pose.orientation
        if dc is not None:
            p = p + dc[0:3]
            q = rotation.boxplus(q, dc[3:6])
        R_wb = rotation.to_matrix(q)
        return CameraModel(
            fx=self.cfg.fx,
            fy=self.cfg.fy,
            cx=self.cfg.cx,
            cy=self.cfg.cy,
            width=self.cfg.width,
            height=self.cfg.height,
            R_wc=(R_wb @ mount_rotation(self.cfg.mount)).T,
            t_w=p + R_wb @ np.asarray(self.cfg.mount.offset),
        )


def _cov_of(state: TrackState | SRTrackState) -> NDArray[np.float64]:
    if isinstance(state, SRTrackState):
        return np.asarray(state.S @ state.S.T)
    return np.asarray(state.P)


def _with_cov(
    state: TrackState | SRTrackState, x: NDArray[np.float64], P: NDArray[np.float64]
) -> tuple[TrackState | SRTrackState, bool]:
    """Rebuild the carrier; returns (state, repaired)."""
    P = 0.5 * (P + P.T)
    if isinstance(state, SRTrackState):
        try:
            return SRTrackState(x=x, S=np.asarray(np.linalg.cholesky(P), dtype=float)), False
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(P)
            floor = 1e-12 * max(float(w[-1]), 1.0)
            S_rep = np.asarray(np.linalg.cholesky((V * np.clip(w, floor, None)) @ V.T), dtype=float)
            return SRTrackState(x=x, S=S_rep), True
    return TrackState(x=x, P=P), False


class ConsiderCamera:
    """Measurement prediction and update for one track seen by one mounted camera."""

    def __init__(self, mounted: MountedCamera, R: NDArray[np.float64], cfg: UKFConfig) -> None:
        self.mounted = mounted
        self.R = R
        self.n_x = cfg.state_dim
        n = self.n_x + POSE_ERR_DIM
        lam = cfg.alpha**2 * (n + cfg.kappa) - n
        self.gamma = float(np.sqrt(n + lam))
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = self.Wm[0] + (1.0 - cfg.alpha**2 + cfg.beta)
        self.repairs = 0

    def _moments(
        self, state: TrackState | SRTrackState, P_xc: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Joint unscented transform through h(x, c): z_bar, S, P_xz, P_cz, P_xx."""
        n_x, n_c = self.n_x, POSE_ERR_DIM
        P_xx = _cov_of(state)
        P_cc = self.mounted.pose.cov
        J = np.block([[P_xx, P_xc], [P_xc.T, P_cc]])
        J = 0.5 * (J + J.T)
        try:
            S_j = np.linalg.cholesky(J)
        except np.linalg.LinAlgError:
            self.repairs += 1
            w, V = np.linalg.eigh(J)
            S_j = np.linalg.cholesky((V * np.clip(w, 1e-12, None)) @ V.T)
        n = n_x + n_c
        mean = np.concatenate([state.x, np.zeros(n_c)])
        sigma = np.vstack([mean, mean + self.gamma * S_j.T, mean - self.gamma * S_j.T])
        m = self.R.shape[0]
        Z = np.zeros((2 * n + 1, m))
        for i in range(2 * n + 1):
            Z[i] = h_camera(sigma[i, :n_x], self.mounted.camera(sigma[i, n_x:]))
        z_bar = self.Wm @ Z
        dz = Z - z_bar
        S = (dz.T * self.Wc) @ dz + self.R
        dx = sigma[:, :n_x] - state.x
        dc = sigma[:, n_x:]
        P_xz = (dx.T * self.Wc) @ dz
        P_cz = (dc.T * self.Wc) @ dz
        return z_bar, S, P_xz, P_cz, P_xx

    def measurement_prediction(
        self, state: TrackState | SRTrackState, P_xc: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(z_pred, S) for gating/PDA — S carries the pose-error contribution."""
        z_bar, S, _, _, _ = self._moments(state, P_xc)
        return z_bar, S

    def update(
        self,
        state: TrackState | SRTrackState,
        P_xc: NDArray[np.float64],
        measurement: NDArray[np.float64],
    ) -> tuple[
        TrackState | SRTrackState, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Returns (state, P_xc, innovation, S)."""
        z_bar, S, P_xz, P_cz, P_xx = self._moments(state, P_xc)
        K = np.linalg.solve(S, P_xz.T).T
        nu = measurement - z_bar
        x_new = state.x + K @ nu
        P_new = P_xx - K @ S @ K.T
        P_xc_new = P_xc - K @ P_cz.T
        new_state, repaired = _with_cov(state, x_new, P_new)
        self.repairs += int(repaired)
        return new_state, P_xc_new, nu, S


def cv_transition(dt: float, state_dim: int = 9) -> NDArray[np.float64]:
    """F of the constant-velocity model (predict P_xc <- F P_xc)."""
    F = np.eye(state_dim)
    F[0:3, 3:6] = dt * np.eye(3)
    return F
