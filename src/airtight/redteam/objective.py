from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from airtight.contracts import EpisodeResult, FleetConfig, SensorCurves, Site, Tactic

EpisodeFn = Callable[[Site, FleetConfig, Tactic, SensorCurves, int, Path], EpisodeResult]


class TacticScore(BaseModel):
    tactic_id: str
    family: str
    n_episodes: int
    miss_rate: float
    mean_margin_s: float
    adversary_score: float


def margin_s(r: EpisodeResult, response_time_s: float) -> float:
    """Seconds of slack before the critical detection point; negative when the alarm was late or never came."""
    if r.t_alarm is None:
        return -response_time_s
    return r.t_cdp - r.t_alarm


def summarize(
    tactic: Tactic, results: Sequence[EpisodeResult], response_time_s: float, margin_weight: float
) -> TacticScore:
    """Adversary score in [0, 1]: miss rate, blended with a continuous near-miss term that orders equal miss rates."""
    misses = np.array([0.0 if r.timely_detected else 1.0 for r in results])
    margins = np.array([margin_s(r, response_time_s) for r in results])
    near_miss = (
        np.clip(1.0 - margins / response_time_s, 0.0, 1.0) if response_time_s > 0 else misses
    )
    score = (1.0 - margin_weight) * misses.mean() + margin_weight * near_miss.mean()
    return TacticScore(
        tactic_id=tactic.id,
        family=tactic.family,
        n_episodes=len(results),
        miss_rate=float(misses.mean()),
        mean_margin_s=float(margins.mean()),
        adversary_score=float(score),
    )


def _run_job(
    args: tuple[EpisodeFn, Site, FleetConfig, Tactic, SensorCurves, int, Path],
) -> tuple[str, int, EpisodeResult]:
    fn, site, fleet, tactic, curves, seed, log_dir = args
    return tactic.id, seed, fn(site, fleet, tactic, curves, seed, log_dir)


def evaluate(
    tactics: Sequence[Tactic],
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    margin_weight: float,
    workers: int = 1,
) -> list[TacticScore]:
    """Run every tactic on every seed, in a process pool when workers > 1, and score each tactic."""
    jobs = [(episode_fn, site, fleet, t, curves, s, log_dir) for t in tactics for s in seeds]
    if workers <= 1:
        outcomes = [_run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_run_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    by_id: dict[str, list[EpisodeResult]] = {t.id: [] for t in tactics}
    for tid, _, res in outcomes:
        by_id[tid].append(res)
    return [summarize(t, by_id[t.id], site.response_time_s, margin_weight) for t in tactics]
