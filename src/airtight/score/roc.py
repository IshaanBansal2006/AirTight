"""Offline ROC and the operating point, pure numpy.

Inputs are plain arrays. peaks has shape (n_seeds, n_tactics): each intrusion episode's
intruder_peak, with the same seeds in every column. quiet is a list of (benign_peaks, hours)
pairs, one per quiet-night run.

The unit of resampling is the seed, never the episode. Episodes that share a seed share a patrol
realisation and a benign world, so the cells of one row are not independent. Every interval here
resamples whole seed rows, with all their tactics together, and whole quiet runs.

The operating point uses no interpolation: it is the lowest candidate threshold whose false alarm
rate is at most the target. Detection is the mean over tactics of each tactic's detected
fraction, so every tactic weighs the same. NEVER_SEEN is never a candidate threshold and never
counts as detected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.sim.constants import NEVER_SEEN, TAU_INVESTIGATE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    Quiet = Sequence[tuple[Array, float]]

FLAG_OK = "ok"
FLAG_FAR_LIMITED = "far_limited"  # already at or under the target at tau_min
FLAG_FAR_FLOOR = "far_floor_above_target"  # no candidate gets under the target
BOOTSTRAP_STREAM = 9


@dataclass(frozen=True)
class OperatingPoint:
    tau: float
    pd: float
    far: float  # false alarms per hour at tau
    flag: str


@dataclass(frozen=True)
class RocGrid:
    tau: Array
    pd: Array
    far: Array


@dataclass(frozen=True)
class Bootstrap:
    pd_ci: tuple[float, float]  # overall pd at each replicate's own operating point
    worst_index: int  # the tactic that is worst in the ORIGINAL data
    worst_pd_ci: tuple[float, float]  # that tactic's pd at each replicate's operating point
    grid_tau: Array  # the roc_grid thresholds, held fixed
    grid_pd_ci: Array  # shape (len(grid_tau), 2)
    n_replicates: int


def _as_peaks(peaks: Array) -> Array:
    arr = np.asarray(peaks, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"peaks must have shape (n_seeds >= 1, n_tactics >= 1), got {arr.shape}")
    return arr


def _benign_and_hours(quiet: Quiet) -> tuple[Array, float]:
    hours = float(sum(h for _, h in quiet))
    if not hours > 0:
        raise ValueError("quiet runs must add up to a positive number of hours")
    arrays = [np.asarray(b, dtype=np.float64).ravel() for b, _ in quiet]
    benign = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)
    return benign, hours


def candidates(peaks: Array, quiet: Quiet, tau_min: float = TAU_INVESTIGATE) -> Array:
    """tau_min plus every distinct intruder or benign peak at or above it, ascending."""
    benign, _ = _benign_and_hours(quiet)
    values = np.concatenate([_as_peaks(peaks).ravel(), benign])
    values = values[(values >= tau_min) & (values > NEVER_SEEN)]
    out: Array = np.unique(np.concatenate([[tau_min], values]))
    return out


def far_curve(quiet: Quiet, taus: Array) -> Array:
    """False alarms per hour at each tau. A benign object counts once if its peak >= tau."""
    benign, hours = _benign_and_hours(quiet)
    taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
    counts = (benign[None, :] >= taus[:, None]).sum(axis=1)
    out: Array = counts / hours
    return out


def pd_by_tactic(peaks: Array, tau: float) -> tuple[Array, int]:
    """Each tactic's fraction of seeds with peak >= tau, and the index of the worst (first on ties)."""
    arr = _as_peaks(peaks)
    detected = (arr >= tau) & (arr > NEVER_SEEN)
    per_tactic: Array = detected.mean(axis=0)
    return per_tactic, int(np.argmin(per_tactic))


def _pd_curve(peaks: Array, taus: Array) -> Array:
    """Overall pd (mean over tactics) at each tau."""
    arr = _as_peaks(peaks)
    detected = (arr[:, :, None] >= taus[None, None, :]) & (arr[:, :, None] > NEVER_SEEN)
    out: Array = detected.mean(axis=0).mean(axis=0)
    return out


def operating_point(
    peaks: Array, quiet: Quiet, far_target: float = 1.0, tau_min: float = TAU_INVESTIGATE
) -> OperatingPoint:
    cands = candidates(peaks, quiet, tau_min)
    far = far_curve(quiet, cands)
    under = np.flatnonzero(far <= far_target)  # far is non-increasing, so these are a suffix
    if under.size == 0:
        tau = float(np.nextafter(cands[-1], np.inf))  # just above everything: nothing is detected
        return OperatingPoint(tau=tau, pd=0.0, far=0.0, flag=FLAG_FAR_FLOOR)
    i = int(under[0])
    tau = float(cands[i])
    per_tactic, _ = pd_by_tactic(peaks, tau)
    flag = FLAG_FAR_LIMITED if i == 0 else FLAG_OK
    return OperatingPoint(tau=tau, pd=float(per_tactic.mean()), far=float(far[i]), flag=flag)


def _grid_taus(
    peaks: Array, quiet: Quiet, n_points: int, far_target: float, tau_min: float
) -> Array:
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    cands = candidates(peaks, quiet, tau_min)
    op = operating_point(peaks, quiet, far_target, tau_min)
    must = np.array([tau_min, op.tau], dtype=np.float64)
    room = max(n_points - 2, 0)
    if len(cands) <= room:
        picked = cands
    else:
        picked = cands[np.unique(np.linspace(0, len(cands) - 1, room).round().astype(int))]
    out: Array = np.unique(np.concatenate([must, picked]))
    return out


def roc_grid(
    peaks: Array,
    quiet: Quiet,
    n_points: int = 40,
    far_target: float = 1.0,
    tau_min: float = TAU_INVESTIGATE,
) -> RocGrid:
    """tau, pd and far on at most n_points candidate thresholds, always with tau_min and the
    operating point among them."""
    taus = _grid_taus(peaks, quiet, n_points, far_target, tau_min)
    return RocGrid(tau=taus, pd=_pd_curve(peaks, taus), far=far_curve(quiet, taus))


def _percentiles(values: Array) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (float(lo), float(hi))


def _replicates(
    arr: Array, quiet: Quiet, n_boot: int, seed: int, far_target: float, tau_min: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], Array]:
    """Draw the replicates. Returns the seed-row multiplicities (B, n_seeds), the quiet-run
    multiplicities (B, n_quiet), and each replicate's pd per tactic at its own operating point
    (B, n_tactics).

    Thresholds are searched on the ORIGINAL candidate set. A replicate's own candidates are a
    subset of it, and no replicate peak lies between the two answers, so its pd is exactly what
    its own candidate set would give. A test checks this against a naive loop.
    """
    n_seeds = arr.shape[0]
    rng = np.random.default_rng([seed, BOOTSTRAP_STREAM])
    cands = candidates(arr, quiet, tau_min)
    row_w = rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=n_boot)
    quiet_w = rng.multinomial(len(quiet), np.full(len(quiet), 1.0 / len(quiet)), size=n_boot)

    hours = np.array([h for _, h in quiet], dtype=np.float64)
    counts = np.stack(
        [
            (np.asarray(b, dtype=np.float64).ravel()[None, :] >= cands[:, None]).sum(axis=1)
            for b, _ in quiet
        ]
    ).astype(np.float64)  # (n_quiet, n_cands)
    rep_far = (quiet_w @ counts) / (quiet_w @ hours)[:, None]  # (B, n_cands)
    under = rep_far <= far_target
    found = under.any(axis=1)
    op_index = np.where(found, under.argmax(axis=1), 0)  # the first candidate under the target

    alive = arr > NEVER_SEEN
    detected = (arr[:, :, None] >= cands[op_index][None, None, :]) & alive[:, :, None]
    per_tactic: Array = np.einsum("bs,stb->bt", row_w, detected.astype(np.float64)) / n_seeds
    per_tactic[~found] = 0.0  # far_floor_above_target: nothing is detected
    return row_w, quiet_w, per_tactic


def bootstrap(
    peaks: Array,
    quiet: Quiet,
    n_boot: int = 500,
    seed: int = 0,
    far_target: float = 1.0,
    tau_min: float = TAU_INVESTIGATE,
    n_points: int = 40,
) -> Bootstrap:
    """Resample seed rows and quiet runs with replacement; recompute the operating point each time.

    Vectorised over replicates. A replicate is a vector of multiplicities (how many times each
    seed row, and each quiet run, was drawn), which is the same thing as resampling with
    replacement. Uses numpy.random.default_rng([seed, 9]).
    """
    arr = _as_peaks(peaks)
    _benign_and_hours(quiet)
    n_seeds = arr.shape[0]
    _, worst = pd_by_tactic(arr, operating_point(arr, quiet, far_target, tau_min).tau)
    row_w, _, per_tactic = _replicates(arr, quiet, n_boot, seed, far_target, tau_min)

    grid_tau = _grid_taus(arr, quiet, n_points, far_target, tau_min)
    detected_grid = (arr[:, :, None] >= grid_tau[None, None, :]) & (arr > NEVER_SEEN)[:, :, None]
    grid_pd = (np.einsum("bs,stg->btg", row_w, detected_grid.astype(np.float64)) / n_seeds).mean(
        axis=1
    )  # (B, n_grid), tau held fixed
    grid_ci: Array = np.percentile(grid_pd, [2.5, 97.5], axis=0).T

    return Bootstrap(
        pd_ci=_percentiles(per_tactic.mean(axis=1)),
        worst_index=worst,
        worst_pd_ci=_percentiles(per_tactic[:, worst]),
        grid_tau=grid_tau,
        grid_pd_ci=grid_ci,
        n_replicates=n_boot,
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n independent trials. For single cells only:
    it assumes independence, which the cells of one seed row do not have."""
    if n == 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
