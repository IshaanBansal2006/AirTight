"""Unscented Kalman Filter.

Sensor-agnostic — observation model h(x) is passed in at update time.
Ref: Wan & Van der Merwe (2000), "The Unscented Kalman Filter for Nonlinear Estimation".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge.types import TrackState

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from airtight.swarm.edge.config import UKFConfig


class UKF:
    def __init__(self, config: UKFConfig) -> None:
        self.cfg = config
        self.n = config.state_dim
        self.Q = np.diag(config.process_noise)
        self.Wm, self.Wc = self._compute_weights()
        self.repairs = 0  # count of PSD repairs triggered (numerical-health telemetry)

    def _compute_weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sigma point weights Wm (mean) and Wc (covariance), shape (2n+1,) each.

        lambda = alpha^2 * (n + kappa) - n
        Wm[0] = lambda / (n + lambda)
        Wc[0] = Wm[0] + (1 - alpha^2 + beta)
        Wm[i] = Wc[i] = 1 / (2(n + lambda))   for i = 1..2n
        """

        self.lam = self.cfg.alpha**2 * (self.n + self.cfg.kappa) - self.n
        Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lam)))
        Wc = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lam)))
        Wm[0] = self.lam / (self.n + self.lam)
        Wc[0] = Wm[0] + (1 - self.cfg.alpha**2 + self.cfg.beta)
        return Wm, Wc

    def generate_sigma_points(self, state: TrackState) -> NDArray[np.float64]:
        """Generate 2n+1 sigma points from state (x, P).

        Returns shape (2n+1, n). Uses Cholesky of (n + lambda) * P.
        """

        sigma = np.zeros((2 * self.n + 1, self.n))
        sigma[0] = state.x
        scaled = (self.n + self.lam) * state.P
        try:
            L = np.linalg.cholesky(scaled)
        except np.linalg.LinAlgError:
            # P went non-PSD (the P - K S K^T shortcut form is not PSD-guaranteed,
            # and strong measurement nonlinearity can overshoot). Repair by
            # eigenvalue flooring: clip negative eigenvalues to a small positive
            # floor and rebuild. Symmetric by construction.
            self.repairs += 1
            w, V = np.linalg.eigh(0.5 * (scaled + scaled.T))
            floor = 1e-9 * max(float(w.max()), 1.0)
            w = np.clip(w, floor, None)
            L = np.linalg.cholesky((V * w) @ V.T)
        # numpy cholesky is LOWER-triangular (L L^T = scaled). The sigma offsets
        # must be vectors s_i with sum s_i s_i^T = scaled — that is the COLUMNS
        # of L, i.e. the rows of L^T. Using rows of L reproduces L^T L instead:
        # right scale, WRONG correlation directions (latent bug found 2026-07-29,
        # regression-tested in tests/test_ukf.py).
        sqrt_P = L.T
        sigma[1 : self.n + 1] = state.x + sqrt_P
        sigma[self.n + 1 :] = state.x - sqrt_P
        return sigma

    def predict(self, state: TrackState) -> TrackState:
        """Predict state forward by dt. Unscented transform through f(x), add Q."""
        sigma = self._f_batch(self.generate_sigma_points(state))
        x_pred = self.Wm @ sigma
        diff = sigma - x_pred
        P_pred = (self.Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
        P_pred = P_pred + self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)  # enforce symmetry against float drift
        return TrackState(x=x_pred, P=P_pred)

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Motion model. Propagate state vector (n,) forward by dt."""
        x_new = np.copy(x)
        x_new[0] += x[3] * self.cfg.dt  # x position
        x_new[1] += x[4] * self.cfg.dt  # y position
        x_new[2] += x[5] * self.cfg.dt  # z position
        return x_new

    def _f_batch(self, sigma: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorised constant-velocity motion; identical to applying ``f`` to each row."""
        out = sigma.copy()
        dt = self.cfg.dt
        out[:, 0] += sigma[:, 3] * dt
        out[:, 1] += sigma[:, 4] * dt
        out[:, 2] += sigma[:, 5] * dt
        return out

    def measurement_prediction(
        self,
        state: TrackState,
        h: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        R: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predicted measurement and innovation covariance (z_pred, S) for gating.

        The unscented transform of the belief through h, without applying any
        measurement — what association (gating / likelihoods) consumes before
        deciding which detection, if any, to fuse.
        """
        sigma = self.generate_sigma_points(state)
        Z = np.vstack([h(s) for s in sigma])
        z_pred = self.Wm @ Z
        diff = Z - z_pred
        S = R + (self.Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
        return z_pred, S

    def update(
        self,
        state: TrackState,
        measurement: NDArray[np.float64],
        h: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        R: NDArray[np.float64],
    ) -> tuple[TrackState, NDArray[np.float64], NDArray[np.float64]]:
        """Incorporate measurement via unscented transform through h(x).

        Args:
            state: predicted state
            measurement: sensor reading, shape (m,)
            h: observation model, state (n,) -> measurement (m,)
            R: measurement noise covariance (m, m)

        Returns:
            (updated_state, innovation, innovation_covariance S)
        """
        sigma = self.generate_sigma_points(state)
        Z = np.vstack([h(s) for s in sigma])
        z_pred = self.Wm @ Z
        diff_z = Z - z_pred
        S = (self.Wc[:, None, None] * diff_z[:, :, None] * diff_z[:, None, :]).sum(axis=0) + R
        diff_x = sigma - state.x
        Pxz = (self.Wc[:, None, None] * diff_x[:, :, None] * diff_z[:, None, :]).sum(axis=0)
        K = np.linalg.solve(S, Pxz.T).T
        # Update state with measurement
        innovation = measurement - z_pred
        x_updated = state.x + K @ innovation
        if self.cfg.covariance_form == "joseph":
            # Joseph-analog for the UKF (no H matrix exists; expressed via Pxz):
            #   P+ = P - K Pxz^T - Pxz K^T + K S K^T
            # Algebraically identical to the shortcut when K = Pxz S^-1 exactly,
            # but quadratic in K — first-order INSENSITIVE to errors in K, which
            # is where the shortcut form loses positive-definiteness.
            P_updated = state.P - K @ Pxz.T - Pxz @ K.T + K @ S @ K.T
        else:  # "shortcut"
            P_updated = state.P - K @ S @ K.T
        P_updated = 0.5 * (P_updated + P_updated.T)  # enforce symmetry against float drift
        # Neither form is PSD-guaranteed under strong nonlinearity (observed live
        # 2026-07-29). Repair HERE, where P is produced: floor negative
        # eigenvalues so the stored covariance stays PSD. Instrumented via
        # self.repairs for the D-B2 strategy bake-off.
        eigvals = np.linalg.eigvalsh(P_updated)
        if eigvals[0] < 0.0:
            self.repairs += 1
            w, V = np.linalg.eigh(P_updated)
            floor = 1e-12 * max(float(w[-1]), 1.0)
            P_updated = (V * np.clip(w, floor, None)) @ V.T
            P_updated = 0.5 * (P_updated + P_updated.T)
        updated_state = TrackState(x=x_updated, P=P_updated)
        return updated_state, innovation, S
