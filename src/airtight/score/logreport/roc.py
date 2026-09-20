from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from airtight.contracts import RocPoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airtight.score.logreport.logs import EpisodeSummary

Z95 = 1.959964


class QuietStats(BaseModel):
    """Benign exposure with no intruder: every benign peak seen, and the hours it took to see them."""

    benign_peaks: list[float] = Field(default_factory=list)
    hours: float
    source: str


def quiet_from_summaries(summaries: Sequence[EpisodeSummary]) -> QuietStats:
    """Fallback when no quiet nights were run: benign peaks inside intrusion episodes, which is little exposure."""
    return QuietStats(
        benign_peaks=[v for s in summaries for v in s.benign_peaks.values()],
        hours=total_hours(summaries),
        source="benign objects inside intrusion episodes",
    )


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def total_hours(summaries: Sequence[EpisodeSummary]) -> float:
    return sum(s.t_end_s for s in summaries) / 3600.0


def pd_at(summaries: Sequence[EpisodeSummary], threshold: float) -> float:
    if not summaries:
        return 0.0
    peaks = np.fromiter(
        (s.intruder_peak_before_cdp for s in summaries), dtype=np.float64, count=len(summaries)
    )
    return float(np.mean(peaks >= threshold))


def far_at(quiet: QuietStats, threshold: float) -> float:
    if quiet.hours <= 0:
        return 0.0
    if not quiet.benign_peaks:
        return 0.0
    peaks = np.asarray(quiet.benign_peaks, dtype=np.float64)
    return float(np.count_nonzero(peaks >= threshold) / quiet.hours)


def _far_curve(quiet: QuietStats, thresholds: Sequence[float]) -> np.ndarray:
    taus = np.asarray(thresholds, dtype=np.float64)
    if quiet.hours <= 0 or taus.size == 0 or not quiet.benign_peaks:
        return np.zeros_like(taus, dtype=np.float64)
    peaks = np.asarray(quiet.benign_peaks, dtype=np.float64)
    return np.count_nonzero(peaks[:, None] >= taus, axis=0).astype(np.float64) / quiet.hours


def roc_curve(
    summaries: Sequence[EpisodeSummary],
    quiet: QuietStats,
    thresholds: Sequence[float] | None = None,
) -> list[RocPoint]:
    thresholds = list(thresholds) if thresholds is not None else threshold_grid(summaries, quiet)
    n = len(summaries)
    if n:
        peaks = np.fromiter(
            (s.intruder_peak_before_cdp for s in summaries), dtype=np.float64, count=n
        )
        hits = np.count_nonzero(peaks[:, None] >= np.asarray(thresholds, dtype=np.float64), axis=0)
    else:
        hits = np.zeros(len(thresholds), dtype=np.int64)
    fars = _far_curve(quiet, thresholds)
    points = []
    for t, k, far in zip(thresholds, hits, fars, strict=True):
        k_int = int(k)
        points.append(
            RocPoint(
                threshold=t,
                pd=k_int / n if n else 0.0,
                pd_ci=wilson(k_int, n),
                far_per_hour=float(far),
            )
        )
    return points


def operating_threshold(quiet: QuietStats, far_target: float, thresholds: Sequence[float]) -> float:
    """Lowest threshold whose false-alarm rate on quiet exposure is at or under the target; the top of the grid if none is."""
    fars = _far_curve(quiet, thresholds)
    under = np.flatnonzero(fars <= far_target)
    if under.size:
        return float(thresholds[int(under[0])])
    return thresholds[-1]


def pd_at_operating_point(
    summaries: Sequence[EpisodeSummary],
    quiet: QuietStats,
    far_target: float,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, tuple[float, float], float]:
    """Pd at the operating point and a bootstrap interval over intrusion episodes at that fixed threshold."""
    rng = rng or np.random.default_rng(0)
    grid = threshold_grid(summaries, quiet)
    tau = operating_threshold(quiet, far_target, grid)
    point = pd_at(summaries, tau)
    if len(summaries) < 2:
        return point, (0.0, 1.0), tau
    hits = np.fromiter(
        (s.intruder_peak_before_cdp >= tau for s in summaries), dtype=float, count=len(summaries)
    )
    idx = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    boots = hits[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, (float(lo), float(hi)), tau


def threshold_grid(
    summaries: Sequence[EpisodeSummary], quiet: QuietStats, step: float = 0.25, floor: float = -5.0
) -> list[float]:
    peaks = [
        s.intruder_peak_before_cdp for s in summaries if math.isfinite(s.intruder_peak_before_cdp)
    ]
    peaks += [v for v in quiet.benign_peaks if math.isfinite(v)]
    top = max(peaks) if peaks else 5.0
    n = int(math.ceil((top - floor) / step)) + 2
    return [floor + i * step for i in range(max(n, 2))]
