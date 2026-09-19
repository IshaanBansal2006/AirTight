"""Filter factory — constructs the estimator selected by config.

`UKFConfig.covariance_form` picks the strategy; consumers call `make_filter`
and `initial_state` instead of naming a filter class, so swapping strategies
(or re-running the D-B2 bake-off) never touches call sites. UKF and
SquareRootUKF share the same predict/update signatures; only the state carrier
differs (P vs its Cholesky factor), which `initial_state` hides.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge.srukf import SquareRootUKF, SRTrackState
from airtight.swarm.edge.types import TrackState
from airtight.swarm.edge.ukf import UKF

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from airtight.swarm.edge.config import UKFConfig


def make_filter(config: UKFConfig) -> UKF | SquareRootUKF:
    """SR-UKF for "srukf" (the default), standard UKF for "shortcut"/"joseph"."""
    if config.covariance_form == "srukf":
        return SquareRootUKF(config)
    return UKF(config)


def initial_state(
    filt: UKF | SquareRootUKF,
    x0: NDArray[np.float64],
    P0: NDArray[np.float64],
) -> TrackState | SRTrackState:
    """Build the filter-appropriate state from a mean and covariance."""
    if isinstance(filt, SquareRootUKF):
        return SRTrackState(x=x0, S=np.linalg.cholesky(P0))
    return TrackState(x=x0, P=P0)


def sqrt_cov_block(state: TrackState | SRTrackState, dim: int = 3) -> NDArray[np.float64]:
    """Lower-triangular Cholesky factor of the LEADING dim x dim block of P.

    The wire format for uncertainty (040 amendment): a factor cannot be
    round-tripped into a non-PSD matrix, unlike a covariance.

    Free for the SR-UKF: because S is lower-triangular, S[:d, d:] is structurally
    zero, so P[:d, :d] = S[:d, :d] @ S[:d, :d].T exactly — the leading block of
    the factor IS the factor of the leading block. No re-factorisation, and
    widening `dim` later costs nothing. This holds ONLY for leading principal
    blocks; the trailing extent block has no such shortcut.
    """
    if isinstance(state, SRTrackState):
        return np.asarray(state.S[:dim, :dim], dtype=float)
    P = np.asarray(state.P, dtype=float)[:dim, :dim]
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        # Standard-form UKF only: P can go indefinite (the failure mode the
        # SR-UKF exists to prevent). Floor the spectrum rather than publish
        # nothing, and make the repair visible instead of silent.
        w, V = np.linalg.eigh(0.5 * (P + P.T))
        repaired = V @ np.diag(np.maximum(w, 1e-12)) @ V.T
        logging.getLogger(__name__).warning(
            "position covariance was not PSD (min eigenvalue %.3e); published a "
            "spectrum-floored factor. Switch covariance_form to 'srukf' to make "
            "this structurally impossible.",
            float(w.min()),
        )
        return np.linalg.cholesky(repaired)
