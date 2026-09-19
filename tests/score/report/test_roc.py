from __future__ import annotations

import math

from airtight.score.report.logs import EpisodeSummary
from airtight.score.report.roc import (
    far_at,
    operating_threshold,
    pd_at_operating_point,
    roc_curve,
    wilson,
)


def _s(
    peak: float,
    benign: dict[str, float] | None = None,
    t_end: float = 600.0,
    seed: int = 0,
    tactic: str = "t",
) -> EpisodeSummary:
    return EpisodeSummary(
        fleet_hash="f",
        tactic_id=tactic,
        family="charging_window",
        seed=seed,
        t_end_s=t_end,
        t_cdp=100.0,
        timely_at_ref=peak >= 4.0,
        intruder_peak_before_cdp=peak,
        benign_peaks=benign or {},
        decoy_peak=None,
        has_score_series=True,
    )


def test_wilson_bounds() -> None:
    lo, hi = wilson(7, 10)
    assert 0.35 < lo < 0.7 < hi < 0.95
    assert wilson(0, 0) == (0.0, 1.0)


def test_roc_is_monotone_in_threshold() -> None:
    summaries = [_s(p, {"b": p - 1.0}, seed=i) for i, p in enumerate([1.0, 2.0, 3.0, 5.0, 6.0])]
    pts = roc_curve(summaries)
    pds = [p.pd for p in pts]
    fars = [p.far_per_hour for p in pts]
    assert pds == sorted(pds, reverse=True) and fars == sorted(fars, reverse=True)


def test_operating_point_picks_lowest_threshold_under_far_target() -> None:
    summaries = [_s(5.0, {f"b{i}": 3.0}, t_end=3600.0, seed=i) for i in range(4)]
    assert far_at(summaries, 2.0) == 1.0 and far_at(summaries, 3.5) == 0.0
    tau = operating_threshold(summaries, far_target=0.5)
    assert 3.0 < tau <= 3.5
    pd, (lo, hi), tau2 = pd_at_operating_point(summaries, far_target=0.5)
    assert pd == 1.0 and tau2 == tau and 0.0 <= lo <= hi <= 1.0


def test_light_log_fallback_scores_by_outcome() -> None:
    summaries = [_s(math.inf, seed=0), _s(-math.inf, seed=1)]
    pts = roc_curve(summaries, thresholds=[0.0, 4.0])
    assert all(p.pd == 0.5 for p in pts)
