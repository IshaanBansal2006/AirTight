from __future__ import annotations

import math

from airtight.score.report.logs import EpisodeSummary
from airtight.score.report.roc import (
    QuietStats,
    far_at,
    operating_threshold,
    pd_at_operating_point,
    quiet_from_summaries,
    roc_curve,
    threshold_grid,
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
    summaries = [_s(p, seed=i) for i, p in enumerate([1.0, 2.0, 3.0, 5.0, 6.0])]
    quiet = QuietStats(benign_peaks=[0.0, 1.0, 2.0, 4.0, 5.0], hours=5.0, source="test")
    pts = roc_curve(summaries, quiet)
    pds = [p.pd for p in pts]
    fars = [p.far_per_hour for p in pts]
    assert pds == sorted(pds, reverse=True) and fars == sorted(fars, reverse=True)


def test_operating_point_uses_quiet_exposure() -> None:
    summaries = [_s(5.0, seed=i) for i in range(4)]
    quiet = QuietStats(benign_peaks=[3.0, 3.0, 3.0, 3.0], hours=4.0, source="test")
    assert far_at(quiet, 2.0) == 1.0 and far_at(quiet, 3.5) == 0.0
    grid = threshold_grid(summaries, quiet)
    tau = operating_threshold(quiet, far_target=0.5, thresholds=grid)
    assert 3.0 < tau <= 3.5
    pd, (lo, hi), tau2 = pd_at_operating_point(summaries, quiet, far_target=0.5)
    assert pd == 1.0 and tau2 == tau and 0.0 <= lo <= hi <= 1.0


def test_fallback_quiet_from_intrusion_episodes() -> None:
    summaries = [
        _s(5.0, {"b1": 2.0}, t_end=1800.0, seed=0),
        _s(5.0, {"b2": 2.5}, t_end=1800.0, seed=1),
    ]
    quiet = quiet_from_summaries(summaries)
    assert (
        quiet.hours == 1.0
        and sorted(quiet.benign_peaks) == [2.0, 2.5]
        and "intrusion" in quiet.source
    )


def test_light_log_fallback_scores_by_outcome() -> None:
    summaries = [_s(math.inf, seed=0), _s(-math.inf, seed=1)]
    quiet = quiet_from_summaries(summaries)
    pts = roc_curve(summaries, quiet, thresholds=[0.0, 4.0])
    assert all(p.pd == 0.5 for p in pts)
