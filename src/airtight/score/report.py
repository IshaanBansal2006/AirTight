"""Build the contract's Report from a sweep, plus a sidecar for what the contract cannot hold.

    python -m airtight.score.report --scenario-dir <dir> --seeds 50 --quiet-seeds 20

Writes data/report.json (the contract's Report, which lane C's charts read) and
data/report_detail.json. Every number comes from the cached sweep cells.

Paired differences. For every configuration except the baseline, paired_vs_baseline holds the
configuration minus the baseline for pd_at_operating_point, worst_tactic_pd and
human_decisions_per_hour, each configuration at its own operating point, on the same seeds. The
intervals come from the seed-level bootstrap applied to both configurations with the SAME
resampled seed indices and the same resampled quiet nights, so the comparison is paired: what
the two configurations share (the benign world, the draw of seeds) cancels. worst_tactic_pd uses
each configuration's own worst tactic in the original data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from airtight.contracts import Conditions, ConfigResult, PairedDelta, Report, RocPoint
from airtight.score import roc
from airtight.score.config_score import ConfigScore, peaks_and_quiet, score_config
from airtight.score.quick import DEFAULT_SEEDS_FILE, split_seeds
from airtight.score.sweep import SweepResult, format_grid_hits, grid_hits, load_inputs, run_sweep
from airtight.sim.constants import ENGINE_IGNORES
from airtight.sim.coverage import coverage_profile, uncovered_s_per_hour
from airtight.sim.episode import official_params

if TYPE_CHECKING:
    import numpy.typing as npt

FAR_TARGET = 1.0
METRIC_PD = "pd_at_operating_point"
METRIC_WORST = "worst_tactic_pd"
METRIC_DECISIONS = "human_decisions_per_hour"


class ConfigDetail(BaseModel):
    operating_threshold: float
    flag: str
    pd_by_tactic: dict[str, float]
    worst_tactic_pd_ci: tuple[float, float]
    raw_alerts_per_hour: float
    quiet_hours: float
    uncovered_phase_ranges: list[tuple[float, float]]  # nobody on duty
    drones_down_phase_ranges: list[tuple[float, float]]  # no drone on duty
    # Seconds per hour with no drone on duty, exact from the clocks. On a site with a guard the
    # contract's coverage_gap_s_per_hour is 0, and this is the number that shows the window.
    drones_down_s_per_hour: float
    tactic_phases_in_uncovered: int
    tactic_phases_in_drones_down: int


class ReportDetail(BaseModel):
    """What the contract's Report cannot hold. Same configuration names, same order."""

    engine_tag: str
    tactic_source: str
    n_tactics: int
    n_seeds: int
    n_quiet_seeds: int
    n_boot: int
    conditions_detail: dict[str, str]  # the long form of the contract's short condition strings
    configs: dict[str, ConfigDetail]


def seed_list_hash(seeds: list[int]) -> str:
    return hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:12]


def _ci(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (float(lo), float(hi))


def paired_deltas(
    score: ConfigScore, reps: roc.Replicates, base: ConfigScore, base_reps: roc.Replicates
) -> list[PairedDelta]:
    """Configuration minus baseline, with paired bootstrap intervals."""
    if not (
        np.array_equal(reps.row_w, base_reps.row_w)
        and np.array_equal(reps.quiet_w, base_reps.quiet_w)
    ):
        raise ValueError("the two configurations were not resampled with the same indices")
    ids, base_ids = list(score.pd_by_tactic), list(base.pd_by_tactic)
    worst, base_worst = ids.index(score.worst_tactic_id), base_ids.index(base.worst_tactic_id)
    rows = (
        (
            METRIC_PD,
            score.pd - base.pd,
            reps.pd_by_tactic.mean(axis=1) - base_reps.pd_by_tactic.mean(axis=1),
        ),
        (
            METRIC_WORST,
            score.worst_tactic_pd - base.worst_tactic_pd,
            reps.pd_by_tactic[:, worst] - base_reps.pd_by_tactic[:, base_worst],
        ),
        (
            METRIC_DECISIONS,
            score.human_decisions_per_hour - base.human_decisions_per_hour,
            reps.far - base_reps.far,
        ),
    )
    return [PairedDelta(metric=m, delta=float(d), ci=_ci(v)) for m, d, v in rows]


def conditions(result: SweepResult) -> Conditions:
    """One short sentence each: lane C's charts print these under an axis. The long versions are
    in the sidecar, see conditions_detail."""
    curves = result.inputs.curves
    jitter = official_params().phase_jitter_s
    return Conditions(
        far_per_hour_operating_point=FAR_TARGET,
        adversary_knowledge=f"knows the patrol policy and charge schedule to within {jitter:g} s, not the patrol's random draws",
        sensor_calibration=f"stub curve {curves.content_hash()} until lane A's calibrated curve arrives",
        detection_model_note="reduced-order detection, truth association, no clutter, 2D footprints",
        seed_list_hash=seed_list_hash(result.seeds),
        n_seeds=len(result.seeds),
    )


def conditions_detail(result: SweepResult) -> dict[str, str]:
    """The long form of each condition, for the sidecar."""
    curves = result.inputs.curves
    jitter = official_params().phase_jitter_s
    return {
        "adversary_knowledge": (
            f"The adversary knows the patrol policy and the charge schedule to within {jitter:g} s: "
            "the intruder enters at phase x reference cycle plus a uniform jitter of that size. It "
            "does not know the patrol's random draws. Tactics are open-loop: the intruder follows "
            "its path whatever the fleet does."
        ),
        "sensor_calibration": (
            "Stub curve until lane A's calibrated curve arrives. "
            f"Source: {curves.source}. Curve hash {curves.content_hash()}."
        ),
        "detection_model_note": (
            "Reduced-order detection: a per-look draw from detection probability by range, with "
            "truth association (every look is credited to the right object), no clutter, and 2D "
            "footprints (a disc or a wedge with a hard range limit). The engine still ignores: "
            + "; ".join(ENGINE_IGNORES)
            + "."
        ),
        "operating_point": (
            "Per configuration: the lowest threshold whose false alarm rate, measured on quiet "
            "nights, is at most the target. No interpolation. Detection is the mean over tactics. "
            "Intervals resample whole seeds and whole quiet nights."
        ),
    }


def build_report(
    result: SweepResult,
    n_boot: int = 500,
    boot_seed: int = 0,
    generated_at: datetime | None = None,
) -> tuple[Report, ReportDetail]:
    inputs = result.inputs
    if inputs.baseline not in inputs.fleets:
        raise ValueError(
            f"baseline {inputs.baseline!r} is not one of the configurations {list(inputs.fleets)}"
        )
    scores: dict[str, ConfigScore] = {}
    reps: dict[str, roc.Replicates] = {}
    for name in inputs.fleets:
        episodes, quiet_runs = result.episodes[name], result.quiet[name]
        scores[name] = score_config(episodes, quiet_runs, FAR_TARGET, n_boot, boot_seed)
        _, peaks, quiet = peaks_and_quiet(episodes, quiet_runs)
        reps[name] = roc.replicates(peaks, quiet, n_boot, boot_seed, FAR_TARGET)

    hits = grid_hits(inputs)
    configs, details = [], {}
    for name, fleet in inputs.fleets.items():
        score = scores[name]
        profile = coverage_profile(inputs.site, fleet, inputs.curves, seed=result.seeds[0])
        configs.append(
            ConfigResult(
                config_name=name,
                fleet_hash=fleet.content_hash(),
                n_episodes=len(result.seeds) * len(inputs.tactics),
                roc=[
                    RocPoint(threshold=r.tau, pd=r.pd, pd_ci=r.pd_ci, far_per_hour=r.far)
                    for r in score.roc
                ],
                pd_at_operating_point=score.pd,
                pd_at_operating_point_ci=score.pd_ci,
                worst_tactic_id=score.worst_tactic_id,
                worst_tactic_pd=score.worst_tactic_pd,
                cost_per_hour=fleet.cost_per_hour(),
                coverage_gap_s_per_hour=uncovered_s_per_hour(profile),
                human_decisions_per_hour=score.human_decisions_per_hour,
                paired_vs_baseline=(
                    []
                    if name == inputs.baseline
                    else paired_deltas(
                        score, reps[name], scores[inputs.baseline], reps[inputs.baseline]
                    )
                ),
            )
        )
        details[name] = ConfigDetail(
            operating_threshold=score.tau,
            flag=score.flag,
            pd_by_tactic=score.pd_by_tactic,
            worst_tactic_pd_ci=score.worst_tactic_pd_ci,
            raw_alerts_per_hour=score.raw_alerts_per_hour,
            quiet_hours=score.quiet_hours,
            uncovered_phase_ranges=hits[name]["uncovered"],
            drones_down_phase_ranges=hits[name]["drones_down"],
            drones_down_s_per_hour=sum(b - a for a, b in hits[name]["drones_down"]) * 3600.0,
            tactic_phases_in_uncovered=hits[name]["uncovered_phases_hit"],
            tactic_phases_in_drones_down=hits[name]["drones_down_phases_hit"],
        )
    report = Report(
        generated_at=generated_at or datetime.now(UTC),
        site_hash=inputs.site.content_hash(),
        baseline_config=inputs.baseline,
        conditions=conditions(result),
        configs=configs,
    )
    report.config(report.baseline_config)  # the contract does not check this, so we do
    detail = ReportDetail(
        engine_tag=result.engine_tag,
        tactic_source=inputs.tactic_source,
        n_tactics=len(inputs.tactics),
        n_seeds=len(result.seeds),
        n_quiet_seeds=len(result.quiet_seeds),
        n_boot=n_boot,
        conditions_detail=conditions_detail(result),
        configs=details,
    )
    return report, detail


def write_report(report: Report, detail: ReportDetail, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path, detail_path = out_dir / "report.json", out_dir / "report_detail.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n")
    detail_path.write_text(detail.model_dump_json(indent=2) + "\n")
    return report_path, detail_path


def write_tactics(result: SweepResult, out_dir: Path) -> Path:
    """Every tactic the sweep used, one contract Tactic file each, named <tactic id>.json, so
    other lanes can draw or rerun them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for tactic in result.inputs.tactics:
        (out_dir / f"{tactic.id}.json").write_text(tactic.model_dump_json(indent=2) + "\n")
    return out_dir


def format_report(report: Report, detail: ReportDetail) -> str:
    lines = [
        f"{'configuration':26s} {'$/h':>5s} {'gap s/h':>7s} {'drones down s/h':>15s} "
        f"{'tau':>6s} {'flag':>12s} "
        f"{'pd [95% interval]':>22s}   {'worst tactic':32s} {'pd':>5s}   "
        f"{'delta pd vs baseline [95%]':>28s}"
    ]
    for c in report.configs:
        d = detail.configs[c.config_name]
        lo, hi = c.pd_at_operating_point_ci
        delta = next((p for p in c.paired_vs_baseline if p.metric == METRIC_PD), None)
        delta_text = (
            "baseline"
            if delta is None
            else f"{delta.delta:+.3f} [{delta.ci[0]:+.3f}, {delta.ci[1]:+.3f}]"
        )
        lines.append(
            f"{c.config_name:26s} {c.cost_per_hour:5.0f} {c.coverage_gap_s_per_hour:7.0f} "
            f"{d.drones_down_s_per_hour:15.0f} "
            f"{d.operating_threshold:6.2f} {d.flag:>12s} "
            f"{c.pd_at_operating_point:6.3f} [{lo:.3f}, {hi:.3f}]   "
            f"{c.worst_tactic_id:32s} {c.worst_tactic_pd:5.2f}   {delta_text:>28s}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--tactics-dir", type=Path, action="append", default=[])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--quiet-seeds", type=int, default=20)
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--sweep-dir", type=Path, default=Path("data/sweep"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--speed-cap", type=float, default=None)
    parser.add_argument("--replays", type=int, default=0, help="export this many failure logs")
    args = parser.parse_args(argv)

    inputs = load_inputs(args.scenario_dir, args.tactics_dir, args.speed_cap)
    seeds, quiet_seeds = split_seeds(args.seeds_file, args.seeds, args.quiet_seeds)
    print(format_grid_hits(inputs))
    result = run_sweep(inputs, seeds, quiet_seeds, args.sweep_dir, args.workers, verbose=True)
    start = time.perf_counter()
    report, detail = build_report(result)
    paths = write_report(report, detail, args.out)
    print(f"report built in {time.perf_counter() - start:.1f} s: {paths[0]} and {paths[1]}")
    print(f"tactic files: {write_tactics(result, args.out / 'report_tactics')}")
    print(format_report(report, detail))
    if args.replays:
        from airtight.score.replays import export_failures

        for row in export_failures(result, detail, args.replays, args.out / "replays"):
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
