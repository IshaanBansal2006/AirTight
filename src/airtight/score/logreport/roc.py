from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from airtight.contracts import RocPoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airtight.score.logreport.logs import EpisodeSummary

Z95 = 1.959964


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
    return float(np.mean([s.intruder_peak_before_cdp >= threshold for s in summaries]))


def far_at(summaries: Sequence[EpisodeSummary], threshold: float) -> float:
    hours = total_hours(summaries)
    if hours <= 0:
        return 0.0
    alarms = sum(1 for s in summaries for v in s.benign_peaks.values() if v >= threshold)
    return alarms / hours


def decisions_per_hour_at(summaries: Sequence[EpisodeSummary], threshold: float) -> float:
    hours = total_hours(summaries)
    if hours <= 0:
        return 0.0
    n = sum(
        (1 if s.intruder_peak_before_cdp >= threshold else 0)
        + sum(1 for v in s.benign_peaks.values() if v >= threshold)
        for s in summaries
    )
    return n / hours


def threshold_grid(
    summaries: Sequence[EpisodeSummary], step: float = 0.25, floor: float = -5.0
) -> list[float]:
    peaks = [
        s.intruder_peak_before_cdp for s in summaries if math.isfinite(s.intruder_peak_before_cdp)
    ]
    peaks += [v for s in summaries for v in s.benign_peaks.values() if math.isfinite(v)]
    top = max(peaks) if peaks else 5.0
    n = int(math.ceil((top - floor) / step)) + 2
    return [floor + i * step for i in range(max(n, 2))]


def roc_curve(
    summaries: Sequence[EpisodeSummary], thresholds: Sequence[float] | None = None
) -> list[RocPoint]:
    thresholds = list(thresholds) if thresholds is not None else threshold_grid(summaries)
    n = len(summaries)
    points = []
    for t in thresholds:
        k = sum(1 for s in summaries if s.intruder_peak_before_cdp >= t)
        points.append(
            RocPoint(
                threshold=t,
                pd=k / n if n else 0.0,
                pd_ci=wilson(k, n),
                far_per_hour=far_at(summaries, t),
            )
        )
    return points


def operating_threshold(
    summaries: Sequence[EpisodeSummary],
    far_target: float,
    thresholds: Sequence[float] | None = None,
) -> float:
    """Lowest threshold whose false-alarm rate is at or under the target; the top of the grid if none is."""
    thresholds = list(thresholds) if thresholds is not None else threshold_grid(summaries)
    for t in thresholds:
        if far_at(summaries, t) <= far_target:
            return t
    return thresholds[-1]


def pd_at_operating_point(
    summaries: Sequence[EpisodeSummary],
    far_target: float,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, tuple[float, float], float]:
    """Pd at the operating point, a bootstrap interval that re-derives the threshold per resample, and the threshold."""
    rng = rng or np.random.default_rng(0)
    grid = threshold_grid(summaries)
    tau = operating_threshold(summaries, far_target, grid)
    point = pd_at(summaries, tau)
    if len(summaries) < 2:
        return point, (0.0, 1.0), tau
    arr = np.array(summaries, dtype=object)
    boots = []
    for _ in range(n_boot):
        sample = list(arr[rng.integers(0, len(arr), size=len(arr))])
        boots.append(pd_at(sample, operating_threshold(sample, far_target, grid)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, (float(lo), float(hi)), tau
