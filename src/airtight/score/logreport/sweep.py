from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from airtight.contracts import (
    Conditions,
    ConfigResult,
    FleetConfig,
    PairedDelta,
    Report,
    SensorCurves,
    Site,
    Tactic,
)
from airtight.score.logreport.logs import EpisodeSummary, summarize_log
from airtight.score.logreport.roc import (
    decisions_per_hour_at,
    pd_at,
    pd_at_operating_point,
    roc_curve,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from airtight.redteam.objective import EpisodeFn

log = logging.getLogger(__name__)


def _job(
    args: tuple[EpisodeFn, Site, FleetConfig, Tactic, SensorCurves, int, Path],
) -> EpisodeSummary:
    fn, site, fleet, tactic, curves, seed, log_dir = args
    result = fn(site, fleet, tactic, curves, seed, log_dir)
    return summarize_log(result.log_path)


def run_config(
    site: Site,
    fleet: FleetConfig,
    tactics: Sequence[Tactic],
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    workers: int = 1,
    prune_logs: bool = False,
) -> list[EpisodeSummary]:
    """Every tactic on every seed with full logs, summarised as they finish; logs optionally deleted after."""
    cfg_dir = log_dir / fleet.name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(episode_fn, site, fleet, t, curves, s, cfg_dir) for t in tactics for s in seeds]
    if workers <= 1:
        out = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    if prune_logs:
        shutil.rmtree(cfg_dir, ignore_errors=True)
    log.info(
        "%s: %d episodes, timely at ref %.2f",
        fleet.name,
        len(out),
        float(np.mean([s.timely_at_ref for s in out])) if out else 0.0,
    )
    return out


def config_result(
    fleet: FleetConfig,
    summaries: Sequence[EpisodeSummary],
    far_target: float,
    rng: np.random.Generator,
) -> tuple[ConfigResult, float]:
    pd, ci, tau = pd_at_operating_point(summaries, far_target, rng=rng)
    by_tactic: dict[str, list[EpisodeSummary]] = {}
    for s in summaries:
        by_tactic.setdefault(s.tactic_id, []).append(s)
    worst_id, worst_pd = min(
        ((tid, pd_at(ss, tau)) for tid, ss in by_tactic.items()), key=lambda p: p[1]
    )
    result = ConfigResult(
        config_name=fleet.name,
        fleet_hash=fleet.content_hash(),
        n_episodes=len(summaries),
        roc=roc_curve(summaries),
        pd_at_operating_point=pd,
        pd_at_operating_point_ci=ci,
        worst_tactic_id=worst_id,
        worst_tactic_pd=worst_pd,
        cost_per_hour=fleet.cost_per_hour(),
        coverage_gap_s_per_hour=0.0,
        human_decisions_per_hour=decisions_per_hour_at(summaries, tau),
    )
    return result, tau


def paired_deltas(
    base: Sequence[EpisodeSummary],
    base_tau: float,
    other: Sequence[EpisodeSummary],
    other_tau: float,
    rng: np.random.Generator,
    n_boot: int = 200,
) -> list[PairedDelta]:
    """Per-episode differences on the shared (tactic, seed) keys; the interval is a bootstrap over pairs."""
    b = {(s.tactic_id, s.seed): s for s in base}
    o = {(s.tactic_id, s.seed): s for s in other}
    keys = sorted(set(b) & set(o))
    if not keys:
        return []
    diffs = np.array(
        [
            (o[k].intruder_peak_before_cdp >= other_tau)
            - (b[k].intruder_peak_before_cdp >= base_tau)
            for k in keys
        ],
        dtype=float,
    )
    boots = [
        float(np.mean(diffs[rng.integers(0, len(diffs), size=len(diffs))])) for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = [
        PairedDelta(
            metric="pd_at_operating_point", delta=float(diffs.mean()), ci=(float(lo), float(hi))
        )
    ]
    d_dec = decisions_per_hour_at(other, other_tau) - decisions_per_hour_at(base, base_tau)
    out.append(PairedDelta(metric="human_decisions_per_hour", delta=d_dec, ci=(d_dec, d_dec)))
    return out


def build_report(
    site: Site,
    per_config: dict[str, tuple[FleetConfig, list[EpisodeSummary]]],
    baseline: str,
    far_target: float,
    seeds: Sequence[int],
    seed_list_hash: str,
    conditions_text: dict[str, str],
    rng: np.random.Generator | None = None,
) -> Report:
    rng = rng or np.random.default_rng(0)
    if baseline not in per_config:
        raise KeyError(f"baseline {baseline!r} not among swept configs {sorted(per_config)}")
    results: dict[str, tuple[ConfigResult, float]] = {
        name: config_result(fleet, ss, far_target, rng) for name, (fleet, ss) in per_config.items()
    }
    base_res, base_tau = results[baseline]
    base_ss = per_config[baseline][1]
    configs = []
    for name, (res, tau) in results.items():
        if name != baseline:
            res = res.model_copy(
                update={
                    "paired_vs_baseline": paired_deltas(
                        base_ss, base_tau, per_config[name][1], tau, rng
                    )
                }
            )
        configs.append(res)
    return Report(
        generated_at=datetime.now(UTC),
        site_hash=site.content_hash(),
        baseline_config=baseline,
        conditions=Conditions(
            far_per_hour_operating_point=far_target,
            adversary_knowledge=conditions_text["adversary_knowledge"],
            sensor_calibration=conditions_text["sensor_calibration"],
            detection_model_note=conditions_text["detection_model_note"],
            seed_list_hash=seed_list_hash,
            n_seeds=len(seeds),
        ),
        configs=configs,
    )


def load_top_tactics(tactics_dir: Path, per_family: int) -> list[Tactic]:
    """The best few per family from a search output directory."""
    from airtight.redteam.search import SearchResult

    out: list[Tactic] = []
    for f in sorted(tactics_dir.glob("top_*.json")):
        res = SearchResult.model_validate_json(f.read_text())
        out.extend(res.tactics[:per_family])
    if not out:
        raise FileNotFoundError(
            f"no top_<family>.json in {tactics_dir}; run airtight-redteam search first"
        )
    return out


def seed_list_hash(seeds: Sequence[int]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest()[:12]
