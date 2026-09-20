from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, cast

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


def light(episode_fn: EpisodeFn) -> EpisodeFn:
    """Ask the runner for header-plus-outcome logs when it supports `full_log`; searches run thousands of episodes."""
    try:
        params = inspect.signature(episode_fn).parameters
    except (TypeError, ValueError):
        return episode_fn
    if "full_log" in params and not isinstance(episode_fn, partial):
        return cast("EpisodeFn", partial(cast("Any", episode_fn), full_log=False))
    return episode_fn


def _run_job(
    args: tuple[EpisodeFn, Site, FleetConfig, Tactic, SensorCurves, int, Path],
) -> tuple[str, int, EpisodeResult]:
    fn, site, fleet, tactic, curves, seed, log_dir = args
    return tactic.id, seed, fn(site, fleet, tactic, curves, seed, log_dir)


def schedule_blind(tactic: Tactic, seed: int) -> Tactic:
    """The same tactic with its entry phase drawn from the seed: the adversary knows the site, not the schedule."""
    phase = float(np.random.default_rng([seed, 7919]).uniform(0.0, 1.0))
    return tactic.model_copy(update={"phase": phase})


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
    randomize_phase: bool = False,
) -> list[TacticScore]:
    """Run every tactic on every seed, in a process pool when workers > 1, and score each tactic."""
    fn = light(episode_fn)
    jobs = [
        (fn, site, fleet, schedule_blind(t, s) if randomize_phase else t, curves, s, log_dir)
        for t in tactics
        for s in seeds
    ]
    if workers <= 1:
        outcomes = [_run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_run_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    by_id: dict[str, list[EpisodeResult]] = {t.id: [] for t in tactics}
    for tid, _, res in outcomes:
        by_id[tid].append(res)
    return [summarize(t, by_id[t.id], site.response_time_s, margin_weight) for t in tactics]


def write_replay_logs(
    tactics: Sequence[Tactic],
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    replay_dir: Path,
) -> list[Path]:
    """Full logs for a handful of tactics, one per seed, for lane A's replay and the failure clips."""
    replay_dir.mkdir(parents=True, exist_ok=True)
    return [
        episode_fn(site, fleet, t, curves, s, replay_dir).log_path for t in tactics for s in seeds
    ]
