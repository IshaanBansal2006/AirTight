"""The fix loop: find a change that costs nothing and raises the worst-tactic detection.

    python -m airtight.score.fix --scenario-dir <dir>

What may change, and nothing else: (a) the drones' charge stagger, as a fraction of even spacing
over the drone cycle; (b) the patrol weight mode; (c) the Go2's charge offset, 0 or half its own
cycle. Every candidate has exactly the baseline's agents and cost per hour.

The weight mode is an engine parameter, not a fleet field, so a candidate is a fleet plus a
parameter override. run_episode picks the override up from AIRTIGHT_WEIGHT_MODE.

The search is cheap on purpose. Each candidate faces the baseline's worst tactics from the
sweep, an even draw from the grid, and the midpoints of ITS OWN gaps for every entry. The
adversary knows the schedule, so a candidate's own gaps must be attacked, or a fix could win by
moving its gap to where nobody looks. Each candidate is scored at its own operating point.

The winner is then confirmed on the full sweep tactic set plus its own gap tactics, on the
sweep's seeds, against the baseline on the same tactics and seeds. If its worst-tactic detection
is not better than the baseline's there, it is not called a fix.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from airtight.contracts import ConfigResult, RocPoint
from airtight.score import roc
from airtight.score.config_score import ConfigScore, peaks_and_quiet, score_config
from airtight.score.quick import DEFAULT_SEEDS_FILE, split_seeds
from airtight.score.report import (
    FAR_TARGET,
    METRIC_WORST,
    ConfigDetail,
    ReportDetail,
    build_report,
    paired_deltas,
    write_report,
    write_tactics,
)
from airtight.score.sweep import (
    SweepInputs,
    SweepResult,
    fleet_gaps,
    gap_phases,
    load_inputs,
    run_sweep,
)
from airtight.sim.constants import NEVER_SEEN
from airtight.sim.coverage import coverage_profile, uncovered_s_per_hour
from airtight.sim.episode import EpisodeParams, official_params
from airtight.sim.runner import ENGINE_ENV, run_episode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airtight.contracts import FleetConfig, PairedDelta, Report, Tactic
    from airtight.sim.episode import EpisodeScores

STAGGER_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHT_MODES = ("asset", "uniform", "band")
GO2_OPTIONS = (False, True)  # offset 0, or half the Go2's own cycle
WEIGHT_MODE_ENV = "AIRTIGHT_WEIGHT_MODE"
N_WORST, N_GRID = 10, 10
SEARCH_SEEDS, SEARCH_QUIET = 30, 10
REPLAY_ENGINE = "v0"


@dataclass(frozen=True)
class Candidate:
    name: str
    fleet: FleetConfig
    weight_mode: str
    stagger_fraction: float
    go2_half_cycle: bool

    def params(self) -> EpisodeParams:
        return dataclasses.replace(official_params(), weight_mode=self.weight_mode)

    @property
    def change(self) -> float:
        """How far from the baseline (no stagger, asset mode, Go2 offset 0). Smaller is simpler."""
        return self.stagger_fraction + (self.weight_mode != "asset") + self.go2_half_cycle


@dataclass(frozen=True)
class CandidateScore:
    candidate: Candidate
    drones_down_s_per_hour: float
    n_tactics: int
    tau: float
    flag: str
    pd: float
    worst_tactic_id: str
    worst_tactic_pd: float


def make_candidates(
    baseline: FleetConfig,
    fractions: Sequence[float] = STAGGER_FRACTIONS,
    modes: Sequence[str] = WEIGHT_MODES,
    go2_options: Sequence[bool] = GO2_OPTIONS,
) -> list[Candidate]:
    """Every combination. Only charge offsets and the weight mode differ from the baseline."""
    drones = [a for a in baseline.agents if a.type == "drone"]
    go2s = [a for a in baseline.agents if a.type == "go2"]
    options = list(go2_options) if go2s else [False]
    out = []
    for fraction in fractions:
        for mode in modes:
            for half in options:
                offsets: dict[str, float] = {}
                for i, drone in enumerate(drones):
                    cycle = drone.endurance_s + drone.charge_time_s
                    offset = fraction * i * cycle / len(drones)
                    if offset > 0:
                        offsets[drone.id] = round(offset, 1)
                if half:
                    for go2 in go2s:
                        offsets[go2.id] = round((go2.endurance_s + go2.charge_time_s) / 2.0, 1)
                name = f"{baseline.name}__s{fraction:g}_{mode}_g{'half' if half else '0'}"
                policy = baseline.charge_policy.model_copy(update={"stagger_offsets_s": offsets})
                fleet = baseline.model_copy(update={"name": name, "charge_policy": policy})
                out.append(Candidate(name, fleet, mode, float(fraction), half))
    return out


def common_tactics(
    inputs: SweepInputs, baseline_pd_by_tactic: dict[str, float], n_worst: int, n_grid: int
) -> list[Tactic]:
    """The baseline's n_worst tactics from the sweep, then n_grid more drawn evenly from the rest."""
    by_id = {t.id: t for t in inputs.tactics}
    order = sorted(baseline_pd_by_tactic, key=lambda t: (baseline_pd_by_tactic[t], t))
    worst = order[:n_worst]
    rest = [t.id for t in inputs.tactics if t.id not in worst]
    step = max(len(rest) / max(n_grid, 1), 1.0)
    drawn = [rest[int(k * step)] for k in range(n_grid) if int(k * step) < len(rest)]
    return [by_id[t] for t in [*worst, *dict.fromkeys(drawn)]]


def own_gap_tactics(inputs: SweepInputs, fleet: FleetConfig, skip_ids: set[str]) -> list[Tactic]:
    """The midpoint of every gap of THIS fleet, for every entry. Ids own-<entry>-<phase>."""
    template = inputs.tactics[0]
    out = []
    for entry in inputs.site.entry_points:
        for phase in gap_phases([fleet], []):
            tactic_id = f"own-{entry.id}-{phase:.4f}"
            if tactic_id in skip_ids:
                continue
            out.append(
                template.model_copy(
                    update={
                        "id": tactic_id,
                        "family": "charging_window",
                        "entry_id": entry.id,
                        "phase": phase,
                        "speed_mps": inputs.speed_cap_mps or template.speed_mps,
                        "waypoints": [inputs.site.asset],
                        "decoy": None,
                        "comms_event": None,
                        "origin": "hand",
                    }
                )
            )
    return out


def _drones_down_s_per_hour(fleet: FleetConfig) -> float:
    return sum(b - a for a, b in fleet_gaps(fleet)["drones_down"]) * 3600.0


def _one_config(
    inputs: SweepInputs, name: str, fleet: FleetConfig, tactics: list[Tactic]
) -> SweepInputs:
    return dataclasses.replace(inputs, fleets={name: fleet}, baseline=name, tactics=tactics)


def run_search(
    inputs: SweepInputs,
    baseline_pd_by_tactic: dict[str, float],
    candidates: Sequence[Candidate],
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    out: Path,
    workers: int | None = None,
    n_worst: int = N_WORST,
    n_grid: int = N_GRID,
    n_boot: int = 100,
    verbose: bool = False,
) -> list[CandidateScore]:
    common = common_tactics(inputs, baseline_pd_by_tactic, n_worst, n_grid)
    rows = []
    for i, cand in enumerate(candidates):
        tactics = [*common, *own_gap_tactics(inputs, cand.fleet, {t.id for t in common})]
        one = _one_config(inputs, cand.name, cand.fleet, tactics)
        result = run_sweep(one, seeds, quiet_seeds, out, workers, params=cand.params())
        score = score_config(
            result.episodes[cand.name], result.quiet[cand.name], FAR_TARGET, n_boot
        )
        rows.append(
            CandidateScore(
                cand,
                _drones_down_s_per_hour(cand.fleet),
                len(tactics),
                score.tau,
                score.flag,
                score.pd,
                score.worst_tactic_id,
                score.worst_tactic_pd,
            )
        )
        if verbose:
            print(
                f"  [{i + 1:2d}/{len(candidates)}] {cand.name}: worst {score.worst_tactic_pd:.3f} "
                f"pd {score.pd:.3f} ({result.n_computed} simulated)",
                flush=True,
            )
    return rows


def choose_fix(rows: Sequence[CandidateScore]) -> CandidateScore:
    """Highest worst-tactic detection; ties by overall detection, then the smallest change."""
    if not rows:
        raise ValueError("choose_fix needs at least one candidate")
    return max(rows, key=lambda r: (r.worst_tactic_pd, r.pd, -r.candidate.change))


@dataclass(frozen=True)
class Confirmation:
    tactics: list[Tactic]
    baseline: SweepResult
    fixed: SweepResult
    baseline_score: ConfigScore
    fixed_score: ConfigScore
    deltas: list[PairedDelta]  # fixed minus baseline, same tactics, same seeds
    is_fix: bool


def confirm(
    inputs: SweepInputs,
    winner: Candidate,
    fixed_name: str,
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    out: Path,
    workers: int | None = None,
    n_boot: int = 500,
) -> Confirmation:
    """Both configurations on the full tactic set plus the winner's own gap tactics."""
    fixed_fleet = winner.fleet.model_copy(update={"name": fixed_name})
    tactics = [
        *inputs.tactics,
        *own_gap_tactics(inputs, fixed_fleet, {t.id for t in inputs.tactics}),
    ]
    base_in = _one_config(inputs, inputs.baseline, inputs.fleets[inputs.baseline], tactics)
    fix_in = _one_config(inputs, fixed_name, fixed_fleet, tactics)
    base = run_sweep(base_in, seeds, quiet_seeds, out, workers)
    fixed = run_sweep(fix_in, seeds, quiet_seeds, out, workers, params=winner.params())

    scores, reps = [], []
    for result, name in ((fixed, fixed_name), (base, inputs.baseline)):
        episodes, quiet = result.episodes[name], result.quiet[name]
        scores.append(score_config(episodes, quiet, FAR_TARGET, n_boot))
        _, peaks, q = peaks_and_quiet(episodes, quiet)
        reps.append(roc.replicates(peaks, q, n_boot, 0, FAR_TARGET))
    deltas = paired_deltas(scores[0], reps[0], scores[1], reps[1])
    worst = next(d for d in deltas if d.metric == METRIC_WORST)
    return Confirmation(tactics, base, fixed, scores[1], scores[0], deltas, worst.delta > 0)


def find_pairs(
    baseline: Sequence[EpisodeScores],
    fixed: Sequence[EpisodeScores],
    tau_baseline: float,
    tau_fixed: float,
) -> list[int]:
    """Seeds that are a miss under the baseline and a timely detection under the fix, each at
    its own operating threshold. Strongest catch first."""
    pairs = []
    for b, f in zip(baseline, fixed, strict=True):
        if b.seed != f.seed:
            raise ValueError(f"episodes are not aligned by seed: {b.seed} against {f.seed}")
        if (
            b.intruder_peak < tau_baseline
            and f.intruder_peak >= tau_fixed
            and f.intruder_peak > NEVER_SEEN
        ):
            pairs.append((f.intruder_peak, f.seed))
    return [seed for _, seed in sorted(pairs, key=lambda p: (-p[0], p[1]))]


def export_pairs(
    inputs: SweepInputs, winner: Candidate, conf: Confirmation, out_dir: Path, n: int = 3
) -> tuple[list[dict[str, Any]], str]:
    """Before-and-after logs for lane A. Returns the index rows and a note on the tactic used."""
    base_name, fixed_name = inputs.baseline, next(iter(conf.fixed.inputs.fleets))
    b_eps, f_eps = conf.baseline.episodes[base_name], conf.fixed.episodes[fixed_name]
    tau_b, tau_f = conf.baseline_score.tau, conf.fixed_score.tau

    tactic_id = conf.baseline_score.worst_tactic_id
    note = f"the baseline's worst tactic, {tactic_id}"
    if not find_pairs(b_eps[tactic_id], f_eps[tactic_id], tau_b, tau_f):
        gains = {
            t: conf.fixed_score.pd_by_tactic[t] - conf.baseline_score.pd_by_tactic[t]
            for t in conf.baseline_score.pd_by_tactic
        }
        for t in sorted(gains, key=lambda t: (-gains[t], t)):
            if find_pairs(b_eps[t], f_eps[t], tau_b, tau_f):
                note = (
                    f"no miss-then-timely seed exists for the baseline's worst tactic {tactic_id}; "
                    f"used {t}, the tactic with the largest paired gain that has one "
                    f"({gains[t]:+.2f})"
                )
                tactic_id = t
                break
    seeds = find_pairs(b_eps[tactic_id], f_eps[tactic_id], tau_b, tau_f)[:n]
    tactic = next(t for t in conf.tactics if t.id == tactic_id)
    by_seed = {
        "baseline": {e.seed: e for e in b_eps[tactic_id]},
        "fixed": {e.seed: e for e in f_eps[tactic_id]},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {k: os.environ.get(k) for k in (ENGINE_ENV, WEIGHT_MODE_ENV)}
    rows = []
    try:
        os.environ[ENGINE_ENV] = REPLAY_ENGINE
        for seed in seeds:
            row: dict[str, Any] = {"tactic_id": tactic_id, "seed": seed}
            for side, fleet, mode, tau in (
                ("baseline", inputs.fleets[base_name], "asset", tau_b),
                ("fixed", conf.fixed.inputs.fleets[fixed_name], winner.weight_mode, tau_f),
            ):
                os.environ[WEIGHT_MODE_ENV] = mode
                outcome = run_episode(
                    inputs.site, fleet, tactic, inputs.curves, seed, out_dir, full_log=True
                )
                peak = by_seed[side][seed].intruder_peak
                row[side] = {
                    "configuration": fleet.name,
                    "weight_mode": mode,
                    "operating_threshold": tau,
                    "intruder_peak_before_cdp": None if peak == NEVER_SEEN else peak,
                    "detected_at_operating_threshold": bool(peak >= tau),
                    "timely_detected_at_tau_ref": outcome.timely_detected,
                    "t_alarm": outcome.t_alarm,
                    "t_cdp": outcome.t_cdp,
                    "log_path": str(outcome.log_path),
                }
            rows.append(row)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    (out_dir / "index.json").write_text(
        json.dumps({"tactic": note, "pairs": rows}, indent=2) + "\n"
    )
    return rows, note


# ---- outputs ---------------------------------------------------------------------------------


class SearchRow(BaseModel):
    name: str
    stagger_fraction: float
    weight_mode: str
    go2_half_cycle: bool
    drones_down_s_per_hour: float
    n_tactics: int
    operating_threshold: float
    flag: str
    pd: float
    worst_tactic_id: str
    worst_tactic_pd: float


class FixDetail(BaseModel):
    baseline: str
    fixed_name: str
    is_fix: bool
    verdict: str
    what_changed: dict[str, Any]
    search_seeds: int
    search_quiet_seeds: int
    confirmation_tactics: int
    confirmation_seeds: int
    baseline_on_confirmation_set: dict[str, Any]
    fixed_on_confirmation_set: dict[str, Any]
    paired_deltas: list[dict[str, Any]]
    search: list[SearchRow]


def search_rows(rows: Sequence[CandidateScore]) -> list[SearchRow]:
    return [
        SearchRow(
            name=r.candidate.name,
            stagger_fraction=r.candidate.stagger_fraction,
            weight_mode=r.candidate.weight_mode,
            go2_half_cycle=r.candidate.go2_half_cycle,
            drones_down_s_per_hour=r.drones_down_s_per_hour,
            n_tactics=r.n_tactics,
            operating_threshold=r.tau,
            flag=r.flag,
            pd=r.pd,
            worst_tactic_id=r.worst_tactic_id,
            worst_tactic_pd=r.worst_tactic_pd,
        )
        for r in rows
    ]


def _summary(score: ConfigScore) -> dict[str, Any]:
    return {
        "operating_threshold": score.tau,
        "flag": score.flag,
        "pd": score.pd,
        "pd_ci": score.pd_ci,
        "worst_tactic_id": score.worst_tactic_id,
        "worst_tactic_pd": score.worst_tactic_pd,
        "worst_tactic_pd_ci": score.worst_tactic_pd_ci,
        "human_decisions_per_hour": score.human_decisions_per_hour,
    }


def fix_detail(
    inputs: SweepInputs,
    winner: Candidate,
    fixed_name: str,
    conf: Confirmation,
    rows: Sequence[CandidateScore],
    n_search_seeds: int,
    n_search_quiet: int,
) -> FixDetail:
    worst = next(d for d in conf.deltas if d.metric == METRIC_WORST)
    if conf.is_fix:
        verdict = (
            f"fix: worst-tactic detection {conf.baseline_score.worst_tactic_pd:.3f} -> "
            f"{conf.fixed_score.worst_tactic_pd:.3f} on the same tactics and seeds, paired delta "
            f"{worst.delta:+.3f} [{worst.ci[0]:+.3f}, {worst.ci[1]:+.3f}]"
            + ("" if worst.ci[0] > 0 else " (the interval includes 0: not significant at this n)")
        )
    else:
        verdict = (
            "NOT a fix: on the confirmation set the search winner's worst-tactic detection "
            f"({conf.fixed_score.worst_tactic_pd:.3f}) is not better than the baseline's "
            f"({conf.baseline_score.worst_tactic_pd:.3f})"
        )
    return FixDetail(
        baseline=inputs.baseline,
        fixed_name=fixed_name,
        is_fix=conf.is_fix,
        verdict=verdict,
        what_changed={
            "stagger_fraction": winner.stagger_fraction,
            "stagger_offsets_s": winner.fleet.charge_policy.stagger_offsets_s,
            "weight_mode": winner.weight_mode,
            "go2_half_cycle": winner.go2_half_cycle,
            "cost_per_hour": winner.fleet.cost_per_hour(),
        },
        search_seeds=n_search_seeds,
        search_quiet_seeds=n_search_quiet,
        confirmation_tactics=len(conf.tactics),
        confirmation_seeds=len(conf.fixed.seeds),
        baseline_on_confirmation_set=_summary(conf.baseline_score),
        fixed_on_confirmation_set=_summary(conf.fixed_score),
        paired_deltas=[d.model_dump() for d in conf.deltas],
        search=search_rows(rows),
    )


def fixed_config_result(
    inputs: SweepInputs, winner: Candidate, fixed_name: str, conf: Confirmation
) -> ConfigResult:
    score, fleet = conf.fixed_score, conf.fixed.inputs.fleets[fixed_name]
    profile = coverage_profile(
        inputs.site, fleet, inputs.curves, winner.params(), seed=conf.fixed.seeds[0]
    )
    return ConfigResult(
        config_name=fixed_name,
        fleet_hash=fleet.content_hash(),
        n_episodes=len(conf.fixed.seeds) * len(conf.tactics),
        roc=[
            RocPoint(threshold=r.tau, pd=r.pd, pd_ci=r.pd_ci, far_per_hour=r.far) for r in score.roc
        ],
        pd_at_operating_point=score.pd,
        pd_at_operating_point_ci=score.pd_ci,
        worst_tactic_id=score.worst_tactic_id,
        worst_tactic_pd=score.worst_tactic_pd,
        cost_per_hour=fleet.cost_per_hour(),
        coverage_gap_s_per_hour=uncovered_s_per_hour(profile),
        human_decisions_per_hour=score.human_decisions_per_hour,
        paired_vs_baseline=conf.deltas,
    )


class ReportDetailWithFix(ReportDetail):
    fix: FixDetail | None = None


def add_fix_to_report(
    report: Report,
    detail: ReportDetail,
    inputs: SweepInputs,
    winner: Candidate,
    fixed_name: str,
    conf: Confirmation,
    fix: FixDetail,
) -> tuple[Report, ReportDetailWithFix]:
    """The sidecar always records the search. The contract Report gains the fixed configuration
    only when it really is a fix. Its paired deltas are against the baseline on the confirmation
    tactic set (the sweep's tactics plus the fix's own gap tactics), which the sidecar states."""
    configs = dict(detail.configs)
    new_report = report
    if conf.is_fix:
        gaps = fleet_gaps(conf.fixed.inputs.fleets[fixed_name])
        phases = sorted({t.phase for t in conf.tactics})
        score = conf.fixed_score
        configs[fixed_name] = ConfigDetail(
            operating_threshold=score.tau,
            flag=score.flag,
            pd_by_tactic=score.pd_by_tactic,
            worst_tactic_pd_ci=score.worst_tactic_pd_ci,
            raw_alerts_per_hour=score.raw_alerts_per_hour,
            quiet_hours=score.quiet_hours,
            uncovered_phase_ranges=gaps["uncovered"],
            drones_down_phase_ranges=gaps["drones_down"],
            drones_down_s_per_hour=_drones_down_s_per_hour(conf.fixed.inputs.fleets[fixed_name]),
            tactic_phases_in_uncovered=sum(
                any(a <= p < b for a, b in gaps["uncovered"]) for p in phases
            ),
            tactic_phases_in_drones_down=sum(
                any(a <= p < b for a, b in gaps["drones_down"]) for p in phases
            ),
        )
        new_report = report.model_copy(
            update={
                "configs": [*report.configs, fixed_config_result(inputs, winner, fixed_name, conf)]
            }
        )
    new_detail = ReportDetailWithFix(**{**detail.model_dump(), "configs": configs, "fix": fix})
    return new_report, new_detail


def write_fixed_fleet(winner: Candidate, fixed_name: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fleet_path, params_path = out_dir / f"{fixed_name}.json", out_dir / f"{fixed_name}.params.json"
    fleet_path.write_text(
        winner.fleet.model_copy(update={"name": fixed_name}).model_dump_json(indent=2) + "\n"
    )
    params_path.write_text(
        json.dumps(
            {"env": {WEIGHT_MODE_ENV: winner.weight_mode, ENGINE_ENV: REPLAY_ENGINE}}, indent=2
        )
        + "\n"
    )
    return fleet_path, params_path


def format_search(rows: Sequence[CandidateScore], winner: CandidateScore) -> str:
    lines = [
        f"{'stagger':>7s} {'mode':8s} {'go2':>5s} {'drones down s/h':>15s} {'tactics':>7s} "
        f"{'tau':>6s} {'pd':>6s}   {'worst tactic':34s} {'worst pd':>8s}"
    ]
    for r in rows:
        c = r.candidate
        mark = "  <- winner" if r is winner else ""
        lines.append(
            f"{c.stagger_fraction:7.2f} {c.weight_mode:8s} {'half' if c.go2_half_cycle else '0':>5s} "
            f"{r.drones_down_s_per_hour:15.0f} {r.n_tactics:7d} {r.tau:6.2f} {r.pd:6.3f}   "
            f"{r.worst_tactic_id:34s} {r.worst_tactic_pd:8.3f}{mark}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--tactics-dir", type=Path, action="append", default=[])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--quiet-seeds", type=int, default=20)
    parser.add_argument("--search-seeds", type=int, default=SEARCH_SEEDS)
    parser.add_argument("--search-quiet-seeds", type=int, default=SEARCH_QUIET)
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--sweep-dir", type=Path, default=Path("data/sweep"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    inputs = load_inputs(args.scenario_dir, args.tactics_dir)
    seeds, quiet_seeds = split_seeds(args.seeds_file, args.seeds, args.quiet_seeds)
    s_seeds, s_quiet = split_seeds(args.seeds_file, args.search_seeds, args.search_quiet_seeds)
    t0 = time.perf_counter()
    result = run_sweep(inputs, seeds, quiet_seeds, args.sweep_dir, args.workers)
    report, detail = build_report(result)
    print(f"sweep and report: {result.n_computed} simulated, {time.perf_counter() - t0:.0f} s")

    t1 = time.perf_counter()
    candidates = make_candidates(inputs.fleets[inputs.baseline])
    base_pd = detail.configs[inputs.baseline].pd_by_tactic
    print(
        f"search: {len(candidates)} candidates, {len(s_seeds)} seeds, {len(s_quiet)} quiet nights each"
    )
    rows = run_search(
        inputs, base_pd, candidates, s_seeds, s_quiet, args.sweep_dir, args.workers, verbose=True
    )
    best = choose_fix(rows)
    print(format_search(rows, best))
    print(f"search wall time {time.perf_counter() - t1:.0f} s; winner {best.candidate.name}")

    t2 = time.perf_counter()
    fixed_name = f"{inputs.baseline}_fixed"
    conf = confirm(
        inputs, best.candidate, fixed_name, seeds, quiet_seeds, args.sweep_dir, args.workers
    )
    fix = fix_detail(inputs, best.candidate, fixed_name, conf, rows, len(s_seeds), len(s_quiet))
    print(
        f"confirmation wall time {time.perf_counter() - t2:.0f} s on {len(conf.tactics)} tactics x {len(seeds)} seeds"
    )
    print(fix.verdict)
    for d in conf.deltas:
        print(f"  {d.metric:26s} {d.delta:+.3f} [{d.ci[0]:+.3f}, {d.ci[1]:+.3f}]")

    report, detail_fix = add_fix_to_report(
        report, detail, inputs, best.candidate, fixed_name, conf, fix
    )
    print("written:", *write_report(report, detail_fix, args.out))
    write_tactics(
        dataclasses.replace(result, inputs=dataclasses.replace(inputs, tactics=conf.tactics)),
        args.out / "report_tactics",
    )
    print("fixed fleet:", *write_fixed_fleet(best.candidate, fixed_name, args.out / "fixed"))
    if conf.is_fix:
        pairs, note = export_pairs(inputs, best.candidate, conf, args.out / "replays" / "pairs")
        print(f"replay pairs: {len(pairs)} for {note}")
    print(f"total wall time {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
