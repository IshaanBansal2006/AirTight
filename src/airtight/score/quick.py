"""Score a few fleets quickly: python -m airtight.score.quick --fleet 2drones --phases 0.2 0.7

For each tactic and each phase it makes a variant with that phase and the id tactic@phase, runs
every variant over the same seeds with official_params(), runs the quiet nights, and prints the
operating point, detection with its interval, the worst tactic, both alert rates and the ROC.

Intrusion seeds are the first N of data/seeds.json and quiet-night seeds are the last M, so the
two never share a seed.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from airtight.score.config_score import score_config
from airtight.score.quiet import run_quiet
from airtight.sim import scenarios
from airtight.sim.episode import official_params, simulate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
    from airtight.score.config_score import ConfigScore
    from airtight.sim.episode import EpisodeScores

    EpisodeJob = tuple[Site, FleetConfig, Tactic, SensorCurves, int]

DEFAULT_SEEDS_FILE = Path(__file__).resolve().parents[3] / "data" / "seeds.json"


def _episode_job(job: EpisodeJob) -> EpisodeScores:
    """One intrusion episode. Top level so a process pool can call it."""
    site, fleet, tactic, curves, seed = job
    return simulate(site, fleet, tactic, curves, seed, official_params())


def split_seeds(path: Path, n_seeds: int, n_quiet: int) -> tuple[list[int], list[int]]:
    seeds = [int(s) for s in json.loads(path.read_text())["seeds"]]
    if n_seeds + n_quiet > len(seeds):
        raise ValueError(
            f"{path} has {len(seeds)} seeds, fewer than {n_seeds} intrusion + {n_quiet} quiet"
        )
    return seeds[:n_seeds], seeds[len(seeds) - n_quiet :]


def phase_variants(
    scenario: str, tactic_names: Sequence[str], phases: Sequence[float]
) -> list[Tactic]:
    """One tactic per (name, phase), with id name@phase."""
    return [
        scenarios.load_tactic(name, scenario).model_copy(
            update={"phase": phase, "id": f"{name}@{phase:g}"}
        )
        for name in tactic_names
        for phase in phases
    ]


def score_fleet(
    scenario: str,
    fleet_name: str,
    tactics: Sequence[Tactic],
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    workers: int | None = None,
) -> ConfigScore:
    site = scenarios.load_site(scenario)
    curves = scenarios.load_sensor_curves(scenario)
    fleet = scenarios.load_fleet(fleet_name, scenario)
    jobs = [(site, fleet, tactic, curves, seed) for tactic in tactics for seed in seeds]
    if workers == 1:
        results = [_episode_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_episode_job, jobs, chunksize=max(1, len(seeds) // 4)))
    by_tactic = {
        tactic.id: results[i * len(seeds) : (i + 1) * len(seeds)]
        for i, tactic in enumerate(tactics)
    }
    quiet = run_quiet(site, fleet, curves, quiet_seeds, workers=workers)
    return score_config(by_tactic, quiet)


def format_score(fleet_name: str, score: ConfigScore) -> str:
    lo, hi = score.pd_ci
    wlo, whi = score.worst_tactic_pd_ci
    lines = [
        f"=== {fleet_name}: {score.n_seeds} seeds per tactic, {score.quiet_hours:.0f} quiet hours",
        f"operating point   tau = {score.tau:.3f}   flag = {score.flag}   "
        f"false alarms = {score.far:.2f} per hour",
        f"overall pd        {score.pd:.3f}   [{lo:.3f}, {hi:.3f}]   (mean over tactics, seeds resampled)",
        f"worst tactic      {score.worst_tactic_id}   pd {score.worst_tactic_pd:.3f}   [{wlo:.3f}, {whi:.3f}]",
        "pd per tactic     " + "   ".join(f"{t} {v:.3f}" for t, v in score.pd_by_tactic.items()),
        f"alerts per hour   {score.human_decisions_per_hour:.2f} at tau (human decisions)   "
        f"{score.raw_alerts_per_hour:.2f} at tau_min (no score filtering)",
        f"{'tau':>8s} {'pd':>6s} {'pd 2.5%':>8s} {'pd 97.5%':>9s} {'far/h':>7s}",
    ]
    for row in score.roc:
        mark = "  <- operating point" if row.tau == score.tau else ""
        lines.append(
            f"{row.tau:8.3f} {row.pd:6.3f} {row.pd_ci[0]:8.3f} {row.pd_ci[1]:9.3f} {row.far:7.2f}{mark}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO)
    parser.add_argument("--fleet", nargs="+", required=True, help="one or more fleet names")
    parser.add_argument("--tactics", nargs="+", default=["jog"])
    parser.add_argument("--phases", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=100, help="intrusion seeds: the first N")
    parser.add_argument("--quiet-seeds", type=int, default=20, help="quiet nights: the last M")
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--workers", type=int, default=None, help="1 runs without a pool")
    args = parser.parse_args(argv)

    seeds, quiet_seeds = split_seeds(args.seeds_file, args.seeds, args.quiet_seeds)
    tactics = phase_variants(args.scenario, args.tactics, args.phases)
    start = time.perf_counter()
    for fleet_name in args.fleet:
        score = score_fleet(args.scenario, fleet_name, tactics, seeds, quiet_seeds, args.workers)
        print(format_score(fleet_name, score))
        print()
    print(f"wall time {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
