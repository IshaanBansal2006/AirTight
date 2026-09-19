"""Quiet nights through a process pool: where false alarm rates come from."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from airtight.sim.episode import official_params, simulate_quiet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airtight.contracts import FleetConfig, SensorCurves, Site
    from airtight.sim.episode import EpisodeParams, QuietScores

    QuietJob = tuple[Site, FleetConfig, SensorCurves, int, EpisodeParams]


def _quiet_job(job: QuietJob) -> QuietScores:
    """One quiet night. Top level so a process pool can call it."""
    site, fleet, curves, seed, params = job
    return simulate_quiet(site, fleet, curves, seed, params=params)


def run_quiet(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    params: EpisodeParams | None = None,
    workers: int | None = None,
) -> list[QuietScores]:
    """One quiet night per seed, in seed order. Each lasts one reference cycle, so every charge
    phase is sampled. params defaults to official_params(). workers=1 runs without a pool."""
    params = official_params() if params is None else params
    jobs = [(site, fleet, curves, int(seed), params) for seed in seeds]
    if workers == 1 or len(jobs) <= 1:
        return [_quiet_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_quiet_job, jobs))
