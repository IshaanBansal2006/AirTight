"""Square-root Unscented Kalman Filter (Van der Merwe & Wan, 2001).

D-B2 bake-off contender vs the standard UKF's shortcut/Joseph forms. Instead of
propagating the covariance P, propagates its lower-triangular Cholesky factor S
(P = S S^T). P then CANNOT go indefinite by construction — the classic failure
mode of the standard form — and sigma points need no per-step Cholesky.

The factor is maintained by three primitives:
  - QR decomposition   (recompose a factor from a set of weighted columns)
  - rank-1 Cholesky update / downdate  (fold in the center sigma point / the
    measurement-update subtraction)
Downdates CAN still fail (the subtraction analog); failures are counted in
`self.repairs` and repaired by an eigenvalue-floored rebuild — same telemetry
as the standard UKF for a fair comparison.

Interface mirrors `UKF` but carries `SRTrackState(x, S)` (the factor, not P).
`P` is available as `S @ S.T` when needed (gating, NEES).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from airtight.swarm.edge.config import UKFConfig


@dataclass
class SRTrackState:
    """Track belief as mean + lower-triangular Cholesky factor of P."""

    x: NDArray[np.float64]  # (n,)
    S: NDArray[np.float64]  # (n, n) lower-triangular, P = S S^T


def _chol_rank1(L: NDArray[np.float64], v: NDArray[np.float64], sign: float) -> NDArray[np.float64]:
    """Rank-1 Cholesky update (sign=+1) / downdate (sign=-1): L L^T ± v v^T.

    Standard hyperbolic-rotation algorithm. Raises LinAlgError if a downdate
    would make the factor indefinite.
    """
    L = L.copy()
    v = v.copy()
    n = v.size
    for k in range(n):
        r2 = L[k, k] ** 2 + sign * v[k] ** 2
        if r2 <= 0.0:
            raise np.linalg.LinAlgError("cholesky downdate: factor went indefinite")
        r = np.sqrt(r2)
        c, s = r / L[k, k], v[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k + 1 :, k] = (L[k + 1 :, k] + sign * s * v[k + 1 :]) / c
            v[k + 1 :] = c * v[k + 1 :] - s * L[k + 1 :, k]
    return L


def _qr_factor(columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lower-triangular S with S S^T = columns @ columns.T, via QR of columns^T.

    columns: (n, m) matrix whose m column vectors' outer-product sum we want.
    QR gives columns^T = Q R with R upper (n x n) -> S = R^T. Column signs are
    flipped so the diagonal is positive (canonical factor).
    """
    _, R = np.linalg.qr(columns.T)
    d = np.sign(np.diag(R))
    d[d == 0.0] = 1.0
    return (R * d[:, None]).T


class SquareRootUKF:
    def __init__(self, config: UKFConfig) -> None:
        self.cfg = config
        self.n = config.state_dim
        self.Sq = np.linalg.cholesky(np.diag(config.process_noise))  # sqrt(Q)
        self.lam = config.alpha**2 * (self.n + config.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)
        n, lam = self.n, self.lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = self.Wm[0] + (1.0 - config.alpha**2 + config.beta)
        self.repairs = 0

    # ------------------------------------------------------------------ sigma
    def generate_sigma_points(self, state: SRTrackState) -> NDArray[np.float64]:
        """(2n+1, n) sigma set from the factor — offsets are gamma * COLUMNS of S."""
        sigma = np.zeros((2 * self.n + 1, self.n))
        sigma[0] = state.x
        offsets = self.gamma * state.S  # columns are the offset vectors
        sigma[1 : self.n + 1] = state.x + offsets.T
        sigma[self.n + 1 :] = state.x - offsets.T
        return sigma

    def f(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Constant-velocity motion model (identical to UKF.f; shared model TBD)."""
        x_new = np.copy(x)
        x_new[0] += x[3] * self.cfg.dt
        x_new[1] += x[4] * self.cfg.dt
        x_new[2] += x[5] * self.cfg.dt
        return x_new

    def _center_fold(
        self, S: NDArray[np.float64], dev0: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fold the Wc[0]-weighted center deviation into a factor (update or downdate)."""
        w0 = self.Wc[0]
        v = np.sqrt(abs(w0)) * dev0
        try:
            return _chol_rank1(S, v, +1.0 if w0 >= 0.0 else -1.0)
        except np.linalg.LinAlgError:
            self.repairs += 1
            P = S @ S.T + np.sign(w0) * np.outer(v, v)
            return self._psd_factor(P)

    def _psd_factor(self, P: NDArray[np.float64]) -> NDArray[np.float64]:
        """Eigenvalue-floored factor rebuild (repair path, counted by callers)."""
        w, V = np.linalg.eigh(0.5 * (P + P.T))
        floor = 1e-12 * max(float(w[-1]), 1.0)
        return np.linalg.cholesky((V * np.clip(w, floor, None)) @ V.T)

    # ---------------------------------------------------------------- predict
    def predict(self, state: SRTrackState) -> SRTrackState:
        sigma = self.generate_sigma_points(state)
        dt = self.cfg.dt
        sigma = sigma.copy()
        sigma[:, 0] += sigma[:, 3] * dt
        sigma[:, 1] += sigma[:, 4] * dt
        sigma[:, 2] += sigma[:, 5] * dt
        x_pred = self.Wm @ sigma
        # Recompose the factor from the non-center deviations + sqrt(Q) columns,
        # then fold in the center deviation.
        devs = (sigma[1:] - x_pred).T * np.sqrt(self.Wc[1])  # (n, 2n)
        S_pred = _qr_factor(np.hstack([devs, self.Sq]))
        S_pred = self._center_fold(S_pred, sigma[0] - x_pred)
        return SRTrackState(x=x_pred, S=S_pred)

    def measurement_prediction(
        self,
        state: SRTrackState,
        h: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        R: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(z_pred, S) for gating — same contract as UKF.measurement_prediction."""
        sigma = self.generate_sigma_points(state)
        Z = np.vstack([h(s) for s in sigma])
        z_pred = self.Wm @ Z
        diff = Z - z_pred
        S = R + (self.Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
        return z_pred, S

    # ----------------------------------------------------------------- update
    def update(
        self,
        state: SRTrackState,
        measurement: NDArray[np.float64],
        h: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        R: NDArray[np.float64],
    ) -> tuple[SRTrackState, NDArray[np.float64], NDArray[np.float64]]:
        """Returns (updated_state, innovation, innovation_covariance) like UKF.update."""
        sigma = self.generate_sigma_points(state)
        m = measurement.shape[0]
        Z = np.zeros((2 * self.n + 1, m))
        for i in range(2 * self.n + 1):
            Z[i] = h(sigma[i])
        z_pred = self.Wm @ Z

        zdevs = (Z[1:] - z_pred).T * np.sqrt(self.Wc[1])  # (m, 2n)
        Sy = _qr_factor(np.hstack([zdevs, np.linalg.cholesky(R)]))
        w0 = self.Wc[0]
        v0 = np.sqrt(abs(w0)) * (Z[0] - z_pred)
        try:
            Sy = _chol_rank1(Sy, v0, +1.0 if w0 >= 0.0 else -1.0)
        except np.linalg.LinAlgError:
            self.repairs += 1
            Sy = self._psd_factor(Sy @ Sy.T + np.sign(w0) * np.outer(v0, v0))

        Pxz = (
            self.Wc[:, None, None] * (sigma - state.x)[:, :, None] * (Z - z_pred)[:, None, :]
        ).sum(axis=0)

        S_inn = Sy @ Sy.T  # innovation covariance (returned for gating/JPDA)
        K = np.linalg.solve(S_inn, Pxz.T).T
        innovation = measurement - z_pred
        x_updated = state.x + K @ innovation

        # Factor downdate: P+ = P - K S_inn K^T, applied as m rank-1 downdates
        # with the columns of U = K Sy. THE step where SR-UKF can still fail.
        U = K @ Sy
        S_new = state.S
        try:
            for j in range(m):
                S_new = _chol_rank1(S_new, U[:, j], -1.0)
        except np.linalg.LinAlgError:
            self.repairs += 1
            S_new = self._psd_factor(state.S @ state.S.T - U @ U.T)
        return SRTrackState(x=x_updated, S=S_new), innovation, S_inn
