from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

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
from airtight.score.logreport.logs import EpisodeSummary, summarize_from_scores, summarize_log
from airtight.score.logreport.roc import (
    QuietStats,
    far_at,
    pd_at,
    pd_at_operating_point,
    quiet_from_summaries,
    roc_curve,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from airtight.redteam.objective import EpisodeFn

log = logging.getLogger(__name__)
DEPLOYED_ALARM_THRESHOLD = 4.0


class ConfigInputs(BaseModel):
    """Everything the report needs for one configuration."""

    fleet: FleetConfig
    summaries: list[EpisodeSummary]
    quiet: QuietStats
    coverage_gap_s_per_hour: float
    summaries_blind: list[EpisodeSummary] = Field(
        default_factory=list,
        description="the same tactics with random entry phases; empty when the pass was skipped",
    )


def _job(
    args: tuple[EpisodeFn, Site, FleetConfig, Tactic, SensorCurves, int, Path],
) -> EpisodeSummary:
    fn, site, fleet, tactic, curves, seed, log_dir = args
    result = fn(site, fleet, tactic, curves, seed, log_dir)
    return summarize_log(result.log_path)


def _blind_job(
    args: tuple[EpisodeFn, Site, FleetConfig, Tactic, SensorCurves, int, Path],
) -> EpisodeSummary:
    """The same tactic with its entry phase drawn from the seed; reported under the original tactic id."""
    from airtight.redteam.objective import schedule_blind

    fn, site, fleet, tactic, curves, seed, log_dir = args
    result = fn(site, fleet, schedule_blind(tactic, seed), curves, seed, log_dir)
    return summarize_log(result.log_path).model_copy(update={"tactic_id": tactic.id})


def _v0_summary_job(
    args: tuple[Site, FleetConfig, Tactic, SensorCurves, int, bool],
) -> EpisodeSummary:
    """Official v0 peaks straight from simulate — no JSONL, no recorder."""
    from airtight.redteam.objective import schedule_blind
    from airtight.sim.episode import official_params, simulate

    site, fleet, tactic, curves, seed, randomize_phase = args
    used = schedule_blind(tactic, seed) if randomize_phase else tactic
    scores = simulate(site, fleet, used, curves, seed, official_params())
    summary = summarize_from_scores(fleet, used, scores)
    return summary.model_copy(update={"tactic_id": tactic.id}) if randomize_phase else summary


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
    randomize_phase: bool = False,
) -> list[EpisodeSummary]:
    """Every tactic on every seed, summarised as they finish.

    The v0 engine with prune_logs (the CLI default) never writes episode files: peaks come
    from EpisodeScores. Stub runs and --keep-logs still go through episode_fn and JSONL.
    """
    engine = os.environ.get("AIRTIGHT_ENGINE", "stub")
    if engine == "v0" and prune_logs:
        jobs = [(site, fleet, t, curves, s, randomize_phase) for t in tactics for s in seeds]
        if workers <= 1:
            out = [_v0_summary_job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                out = list(
                    pool.map(_v0_summary_job, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
                )
    else:
        cfg_dir = log_dir / (fleet.name + ("_blind" if randomize_phase else ""))
        cfg_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(episode_fn, site, fleet, t, curves, s, cfg_dir) for t in tactics for s in seeds]
        job = _blind_job if randomize_phase else _job
        if workers <= 1:
            out = [job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                out = list(pool.map(job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
        if prune_logs:
            shutil.rmtree(cfg_dir, ignore_errors=True)
    log.info(
        "%s: %d episodes, timely at ref %.2f",
        fleet.name,
        len(out),
        float(np.mean([s.timely_at_ref for s in out])) if out else 0.0,
    )
    return out


def run_quiet_nights(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    quiet_seeds: Sequence[int],
    workers: int = 1,
) -> QuietStats:
    """Benign-only exposure through the engine's quiet nights; one reference charge cycle per seed."""
    from airtight.score.quiet import run_quiet

    nights = run_quiet(site, fleet, curves, list(quiet_seeds), workers=max(1, workers))
    peaks = [v for night in nights for v in night.benign_peaks.values()]
    hours = float(sum(night.sim_hours for night in nights))
    return QuietStats(
        benign_peaks=peaks,
        hours=hours,
        source=f"{len(nights)} quiet nights of one charge cycle each",
    )


def coverage_gap(site: Site, fleet: FleetConfig, curves: SensorCurves) -> float:
    """Seconds per hour with nobody patrolling, from the engine's coverage profile over one charge cycle."""
    from airtight.sim.coverage import coverage_profile, uncovered_s_per_hour

    return float(uncovered_s_per_hour(coverage_profile(site, fleet, curves)))


def config_result(
    inputs: ConfigInputs, far_target: float, rng: np.random.Generator
) -> tuple[ConfigResult, float]:
    fleet, summaries, quiet = inputs.fleet, inputs.summaries, inputs.quiet
    pd, ci, tau = pd_at_operating_point(summaries, quiet, far_target, rng=rng)
    by_tactic: dict[str, list[EpisodeSummary]] = {}
    for s in summaries:
        by_tactic.setdefault(s.tactic_id, []).append(s)
    worst_id, worst_pd = min(
        ((tid, pd_at(ss, tau)) for tid, ss in by_tactic.items()), key=lambda p: p[1]
    )
    blind: float | None = None
    if inputs.summaries_blind:
        by_tactic_blind: dict[str, list[EpisodeSummary]] = {}
        for b in inputs.summaries_blind:
            by_tactic_blind.setdefault(b.tactic_id, []).append(b)
        blind = min(pd_at(ss, tau) for ss in by_tactic_blind.values())
    result = ConfigResult(
        config_name=fleet.name,
        fleet_hash=fleet.content_hash(),
        n_episodes=len(summaries),
        roc=roc_curve(summaries, quiet),
        pd_at_operating_point=pd,
        pd_at_operating_point_ci=ci,
        worst_tactic_id=worst_id,
        worst_tactic_pd=worst_pd,
        worst_tactic_pd_schedule_blind=blind,
        cost_per_hour=fleet.cost_per_hour(),
        coverage_gap_s_per_hour=inputs.coverage_gap_s_per_hour,
        human_decisions_per_hour=far_at(quiet, DEPLOYED_ALARM_THRESHOLD),
    )
    return result, tau


def paired_deltas(
    base: ConfigInputs,
    base_tau: float,
    other: ConfigInputs,
    other_tau: float,
    rng: np.random.Generator,
    n_boot: int = 200,
) -> list[PairedDelta]:
    """Per-episode differences on the shared (tactic, seed) keys; the interval is a bootstrap over pairs."""
    b = {(s.tactic_id, s.seed): s for s in base.summaries}
    o = {(s.tactic_id, s.seed): s for s in other.summaries}
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
    d_dec = far_at(other.quiet, DEPLOYED_ALARM_THRESHOLD) - far_at(
        base.quiet, DEPLOYED_ALARM_THRESHOLD
    )
    out.append(PairedDelta(metric="human_decisions_per_hour", delta=d_dec, ci=(d_dec, d_dec)))
    d_gap = other.coverage_gap_s_per_hour - base.coverage_gap_s_per_hour
    out.append(PairedDelta(metric="coverage_gap_s_per_hour", delta=d_gap, ci=(d_gap, d_gap)))
    return out


def build_report(
    site: Site,
    per_config: dict[str, ConfigInputs],
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
    results = {name: config_result(inp, far_target, rng) for name, inp in per_config.items()}
    base_res, base_tau = results[baseline]
    configs = []
    for name, (res, tau) in results.items():
        if name != baseline:
            res = res.model_copy(
                update={
                    "paired_vs_baseline": paired_deltas(
                        per_config[baseline], base_tau, per_config[name], tau, rng
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


def inputs_without_quiet(fleet: FleetConfig, summaries: list[EpisodeSummary]) -> ConfigInputs:
    """For tests and the stub engine: false alarms from intrusion episodes, coverage gap unknown."""
    return ConfigInputs(
        fleet=fleet,
        summaries=summaries,
        quiet=quiet_from_summaries(summaries),
        coverage_gap_s_per_hour=0.0,
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
    return hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest()[:12]
