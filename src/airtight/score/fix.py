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

The winner is then scored on the full sweep tactic set plus its own gap tactics, on the
sweep's seeds, against the baseline on the same tactics and seeds. Those numbers are IN-SAMPLE:
the search chose its winner on some of those seeds, so they flatter it (the winner's curse).

The verdict therefore comes from a held-out confirmation: seeds the search never saw, as
intrusion seeds or as quiet nights. The worst tactic of each configuration is chosen on the
in-sample data and then scored on the held-out seeds, so the choice of tactic is not
contaminated either. The fix is "confirmed" only if the held-out paired interval for
worst-tactic detection excludes 0; otherwise it stays a "candidate" and is kept out of the
contract report. The same held-out seeds also take the fix's ingredients apart.
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
    PHASE_DEDUP,
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


def own_gap_tactics(
    inputs: SweepInputs, fleet: FleetConfig, existing: Sequence[Tactic]
) -> list[Tactic]:
    """The midpoint of every gap of THIS fleet, for every entry. Ids own-<entry>-<phase>.

    A midpoint is skipped when existing already holds the same attack: a straight line from that
    entry at the same speed, within PHASE_DEDUP of the phase. Two fleets with the same drone
    schedule have the same gaps, so without this the same episode would be counted twice under
    two ids, which never moves a worst-tactic number but double-weights the mean.
    """
    speed = inputs.speed_cap_mps or inputs.tactics[0].speed_mps
    straight = [inputs.site.asset]
    have = [
        (t.entry_id, t.phase)
        for t in existing
        if t.waypoints == straight and t.speed_mps == speed and t.decoy is None
    ]
    out = []
    for entry in inputs.site.entry_points:
        for phase in gap_phases([fleet], []):
            if any(e == entry.id and abs(ph - phase) < PHASE_DEDUP for e, ph in have):
                continue
            have.append((entry.id, phase))
            out.append(
                inputs.tactics[0].model_copy(
                    update={
                        "id": f"own-{entry.id}-{phase:.4f}",
                        "family": "charging_window",
                        "entry_id": entry.id,
                        "phase": phase,
                        "speed_mps": speed,
                        "waypoints": straight,
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
        tactics = [*common, *own_gap_tactics(inputs, cand.fleet, common)]
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
        *own_gap_tactics(inputs, fixed_fleet, inputs.tactics),
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


# ---- held-out confirmation -------------------------------------------------------------------

CONFIRMED, CANDIDATE = "confirmed", "candidate"


def held_out_split(
    seeds_file: Path,
    n_in_sample: int,
    n_in_sample_quiet: int,
    n_search_quiet: int,
    n_held_out: int | None = None,
) -> tuple[list[int], list[int]]:
    """Seeds the search never saw: (intrusion seeds, quiet seeds).

    The committed list is used front to back for intrusions and back to front for quiet nights.
    The search used a prefix of the in-sample intrusion seeds and the last n_search_quiet quiet
    seeds. Held-out quiet seeds are the in-sample quiet seeds the search did not use; held-out
    intrusion seeds are everything between the in-sample intrusion seeds and the quiet seeds.
    Raises if asked for more than that, because the only way to get more is to overlap.
    """
    seeds = [int(x) for x in json.loads(seeds_file.read_text())["seeds"]]
    quiet_start = len(seeds) - n_in_sample_quiet
    quiet = seeds[quiet_start : len(seeds) - n_search_quiet]
    available = seeds[n_in_sample:quiet_start]
    if not quiet:
        raise ValueError("the search used every quiet seed, so none is left to hold out")
    if n_held_out is not None and n_held_out > len(available):
        raise ValueError(
            f"asked for {n_held_out} held-out intrusion seeds but only {len(available)} exist that "
            f"the search never saw, as intrusion or quiet seeds ({seeds_file} has {len(seeds)})"
        )
    intrusion = available if n_held_out is None else available[:n_held_out]
    return intrusion, quiet


def assert_disjoint(
    held_out: Sequence[int], held_out_quiet: Sequence[int], seen: Sequence[int]
) -> None:
    """Raise if a held-out seed was seen by the search, or is used twice within the hold-out."""
    clash = (set(held_out) | set(held_out_quiet)) & set(seen)
    if clash:
        raise ValueError(f"held-out seeds overlap seeds the search saw: {sorted(clash)[:5]}")
    both = set(held_out) & set(held_out_quiet)
    if both:
        raise ValueError(f"seeds used for both intrusion and quiet nights: {sorted(both)[:5]}")


def verdict_for(worst_delta: PairedDelta) -> str:
    """Confirmed only if the held-out paired interval for worst-tactic detection excludes 0."""
    return CONFIRMED if worst_delta.ci[0] > 0 else CANDIDATE


@dataclass(frozen=True)
class Scored:
    label: str
    fleet: FleetConfig
    weight_mode: str
    result: SweepResult
    score: ConfigScore
    reps: roc.Replicates


def score_one(
    inputs: SweepInputs,
    label: str,
    fleet: FleetConfig,
    weight_mode: str,
    tactics: list[Tactic],
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    out: Path,
    workers: int | None = None,
    n_boot: int = 500,
) -> Scored:
    """One configuration with one weight mode on a given tactic set and seeds, from cache."""
    params = dataclasses.replace(official_params(), weight_mode=weight_mode)
    one = _one_config(inputs, fleet.name, fleet, tactics)
    result = run_sweep(one, seeds, quiet_seeds, out, workers, params=params)
    episodes, quiet = result.episodes[fleet.name], result.quiet[fleet.name]
    score = score_config(episodes, quiet, FAR_TARGET, n_boot)
    _, peaks, q = peaks_and_quiet(episodes, quiet)
    return Scored(
        label, fleet, weight_mode, result, score, roc.replicates(peaks, q, n_boot, 0, FAR_TARGET)
    )


@dataclass(frozen=True)
class HeldOut:
    seeds: list[int]
    quiet_seeds: list[int]
    tactics: list[Tactic]
    baseline: Scored
    fixed: Scored
    in_sample_worst: tuple[str, str]  # (fixed, baseline): chosen on the in-sample data
    deltas: list[PairedDelta]  # worst_tactic_pd uses the in-sample-chosen tactics
    deltas_held_out_worst: list[PairedDelta]  # worst chosen on the held-out data itself
    verdict: str


def held_out_confirmation(
    inputs: SweepInputs,
    winner: Candidate,
    conf: Confirmation,
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    seen_by_search: Sequence[int],
    out: Path,
    workers: int | None = None,
    n_boot: int = 500,
) -> HeldOut:
    """Baseline and fix on the confirmation tactic set, on seeds the search never saw."""
    assert_disjoint(seeds, quiet_seeds, seen_by_search)
    fixed_fleet = next(iter(conf.fixed.inputs.fleets.values()))
    base = score_one(
        inputs,
        "baseline",
        inputs.fleets[inputs.baseline],
        "asset",
        conf.tactics,
        seeds,
        quiet_seeds,
        out,
        workers,
        n_boot,
    )
    fixed = score_one(
        inputs,
        "both (the fix)",
        fixed_fleet,
        winner.weight_mode,
        conf.tactics,
        seeds,
        quiet_seeds,
        out,
        workers,
        n_boot,
    )
    chosen = (conf.fixed_score.worst_tactic_id, conf.baseline_score.worst_tactic_id)
    deltas = paired_deltas(fixed.score, fixed.reps, base.score, base.reps, worst_ids=chosen)
    own = paired_deltas(fixed.score, fixed.reps, base.score, base.reps)
    worst = next(d for d in deltas if d.metric == METRIC_WORST)
    return HeldOut(
        list(seeds),
        list(quiet_seeds),
        conf.tactics,
        base,
        fixed,
        chosen,
        deltas,
        own,
        verdict_for(worst),
    )


def ingredients(
    inputs: SweepInputs,
    winner: Candidate,
    held: HeldOut,
    out: Path,
    workers: int | None = None,
    n_boot: int = 500,
) -> list[tuple[Scored, list[PairedDelta], float, PairedDelta]]:
    """The fix taken apart on the held-out seeds and tactics: baseline, the weight mode alone,
    the charge offsets alone, and both. Each row: the scored configuration, its paired deltas
    against the baseline (worst chosen on the held-out data), its detection on the BASELINE's
    in-sample worst tactic, and that number's paired delta. The last row IS the fix: the same
    object held-out confirmation scored, so its numbers are equal exactly."""
    base_fleet = inputs.fleets[inputs.baseline]
    offsets_only = held.fixed.fleet.model_copy(
        update={"name": f"{held.fixed.fleet.name}__offsets_only"}
    )
    rows = [
        held.baseline,
        score_one(
            inputs,
            f"{winner.weight_mode} weight only",
            base_fleet,
            winner.weight_mode,
            held.tactics,
            held.seeds,
            held.quiet_seeds,
            out,
            workers,
            n_boot,
        ),
        score_one(
            inputs,
            "charge offsets only",
            offsets_only,
            "asset",
            held.tactics,
            held.seeds,
            held.quiet_seeds,
            out,
            workers,
            n_boot,
        ),
        held.fixed,
    ]
    target = held.in_sample_worst[1]
    out_rows = []
    for row in rows:
        deltas = paired_deltas(row.score, row.reps, held.baseline.score, held.baseline.reps)
        on_target = paired_deltas(
            row.score, row.reps, held.baseline.score, held.baseline.reps, worst_ids=(target, target)
        )
        out_rows.append(
            (
                row,
                deltas,
                row.score.pd_by_tactic[target],
                next(d for d in on_target if d.metric == METRIC_WORST),
            )
        )
    return out_rows


def apply_policy(fleet: FleetConfig, winner: Candidate) -> FleetConfig:
    """The winning policy on another fleet: its drones keep their own stagger, every Go2 gets
    the winner's Go2 offset rule. The weight mode travels separately, as a parameter."""
    offsets = dict(fleet.charge_policy.stagger_offsets_s)
    for agent in fleet.agents:
        if agent.type == "go2":
            half = round((agent.endurance_s + agent.charge_time_s) / 2.0, 1)
            if winner.go2_half_cycle:
                offsets[agent.id] = half
            else:
                offsets.pop(agent.id, None)
    policy = fleet.charge_policy.model_copy(update={"stagger_offsets_s": offsets})
    return fleet.model_copy(update={"name": f"{fleet.name}_policy", "charge_policy": policy})


def policy_by_fleet(
    inputs: SweepInputs,
    winner: Candidate,
    fleet_names: Sequence[str],
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    out: Path,
    workers: int | None = None,
    n_boot: int = 500,
    fixed_name: str | None = None,
) -> list[tuple[str, Scored, Scored, list[PairedDelta]]]:
    """Naive against fixed policy for each named fleet, same tactics and seeds, each attacked
    inside its own gaps. Returns (fleet name, naive, with the policy, paired deltas). For the
    baseline itself the policy fleet is the fix, under fixed_name, so its cached cells are reused."""
    rows = []
    for name in fleet_names:
        naive_fleet = inputs.fleets[name]
        if name == inputs.baseline and fixed_name is not None:
            policy_fleet = winner.fleet.model_copy(update={"name": fixed_name})
        else:
            policy_fleet = apply_policy(naive_fleet, winner)
        tactics = [*inputs.tactics, *own_gap_tactics(inputs, policy_fleet, inputs.tactics)]
        tactics += own_gap_tactics(inputs, naive_fleet, tactics)
        naive = score_one(
            inputs, "naive", naive_fleet, "asset", tactics, seeds, quiet_seeds, out, workers, n_boot
        )
        fixed = score_one(
            inputs,
            "policy",
            policy_fleet,
            winner.weight_mode,
            tactics,
            seeds,
            quiet_seeds,
            out,
            workers,
            n_boot,
        )
        rows.append(
            (name, naive, fixed, paired_deltas(fixed.score, fixed.reps, naive.score, naive.reps))
        )
    return rows


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
    # is_fix, verdict and the three fields above are IN-SAMPLE: the search chose its winner on
    # some of those seeds. status and held_out are what count.
    status: str = "in-sample only"  # "confirmed", "candidate", or "in-sample only"
    held_out: dict[str, Any] | None = None
    ingredients: list[dict[str, Any]] = []
    policy_by_fleet: list[dict[str, Any]] = []


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


def _delta_rows(deltas: Sequence[PairedDelta]) -> list[dict[str, Any]]:
    return [d.model_dump() for d in deltas]


def held_out_detail(held: HeldOut) -> dict[str, Any]:
    fixed_worst, base_worst = held.in_sample_worst
    base_pd, fixed_pd = held.baseline.score.pd_by_tactic, held.fixed.score.pd_by_tactic
    return {
        "verdict": held.verdict,
        "rule": "confirmed only if the paired interval for worst-tactic detection excludes 0",
        "n_seeds": len(held.seeds),
        "n_quiet_seeds": len(held.quiet_seeds),
        "n_tactics": len(held.tactics),
        "seeds_never_seen_by_the_search": True,
        "worst_tactic_chosen_in_sample": {
            "baseline": {"tactic": base_worst, "held_out_pd": base_pd[base_worst]},
            "fixed": {"tactic": fixed_worst, "held_out_pd": fixed_pd[fixed_worst]},
        },
        "paired_deltas": _delta_rows(held.deltas),
        "baseline": _summary(held.baseline.score),
        "fixed": _summary(held.fixed.score),
        "paired_deltas_with_worst_chosen_on_held_out_data": _delta_rows(held.deltas_held_out_worst),
    }


def ingredient_rows(
    rows: Sequence[tuple[Scored, list[PairedDelta], float, PairedDelta]],
) -> list[dict[str, Any]]:
    return [
        {
            "label": scored.label,
            "weight_mode": scored.weight_mode,
            "stagger_offsets_s": scored.fleet.charge_policy.stagger_offsets_s,
            **_summary(scored.score),
            "paired_deltas_vs_baseline": _delta_rows(deltas),
            "pd_on_baselines_in_sample_worst_tactic": on_target,
            "paired_delta_on_that_tactic": target_delta.model_dump(),
        }
        for scored, deltas, on_target, target_delta in rows
    ]


def policy_rows(
    rows: Sequence[tuple[str, Scored, Scored, list[PairedDelta]]],
) -> list[dict[str, Any]]:
    return [
        {
            "fleet": name,
            "n_drones": sum(a.type == "drone" for a in naive.fleet.agents),
            "cost_per_hour": naive.fleet.cost_per_hour(),
            "cost_per_hour_with_policy": fixed.fleet.cost_per_hour(),
            "n_tactics": len(naive.score.pd_by_tactic),
            "naive": _summary(naive.score),
            "policy": _summary(fixed.score),
            "paired_deltas": _delta_rows(deltas),
        }
        for name, naive, fixed, deltas in rows
    ]


def fixed_config_result(
    inputs: SweepInputs,
    winner: Candidate,
    fixed_name: str,
    scored: Scored,
    deltas: list[PairedDelta],
) -> ConfigResult:
    """The fix as a contract ConfigResult, from the HELD-OUT run. Its paired deltas are against
    the baseline on the same held-out seeds and tactics."""
    score, fleet = scored.score, scored.fleet
    profile = coverage_profile(
        inputs.site, fleet, inputs.curves, winner.params(), seed=scored.result.seeds[0]
    )
    return ConfigResult(
        config_name=fixed_name,
        fleet_hash=fleet.content_hash(),
        n_episodes=len(scored.result.seeds) * len(score.pd_by_tactic),
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
        paired_vs_baseline=deltas,
    )


class ReportDetailWithFix(ReportDetail):
    fix: FixDetail | None = None


def add_fix_to_report(
    report: Report,
    detail: ReportDetail,
    inputs: SweepInputs,
    winner: Candidate,
    fixed_name: str,
    fix: FixDetail,
    held: HeldOut | None = None,
) -> tuple[Report, ReportDetailWithFix]:
    """The sidecar always records the search and whatever confirmation exists. The contract
    Report gains the fixed configuration only when the held-out verdict is "confirmed", and then
    with the held-out numbers. A candidate, or a fix with no held-out run, stays in the sidecar."""
    configs = dict(detail.configs)
    new_report = report
    status = "in-sample only" if held is None else held.verdict
    if held is not None and held.verdict == CONFIRMED:
        fleet, score = held.fixed.fleet, held.fixed.score
        gaps = fleet_gaps(fleet)
        phases = sorted({t.phase for t in held.tactics})
        configs[fixed_name] = ConfigDetail(
            operating_threshold=score.tau,
            flag=score.flag,
            pd_by_tactic=score.pd_by_tactic,
            worst_tactic_pd_ci=score.worst_tactic_pd_ci,
            raw_alerts_per_hour=score.raw_alerts_per_hour,
            quiet_hours=score.quiet_hours,
            uncovered_phase_ranges=gaps["uncovered"],
            drones_down_phase_ranges=gaps["drones_down"],
            drones_down_s_per_hour=_drones_down_s_per_hour(fleet),
            tactic_phases_in_uncovered=sum(
                any(a <= p < b for a, b in gaps["uncovered"]) for p in phases
            ),
            tactic_phases_in_drones_down=sum(
                any(a <= p < b for a, b in gaps["drones_down"]) for p in phases
            ),
        )
        result = fixed_config_result(inputs, winner, fixed_name, held.fixed, held.deltas)
        new_report = report.model_copy(update={"configs": [*report.configs, result]})
    fix = fix.model_copy(
        update={"status": status, "held_out": None if held is None else held_out_detail(held)}
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


def _ci(d: PairedDelta) -> str:
    return f"{d.delta:+.3f} [{d.ci[0]:+.3f}, {d.ci[1]:+.3f}]"


def format_ingredients(
    rows: Sequence[tuple[Scored, list[PairedDelta], float, PairedDelta]],
) -> str:
    lines = [
        f"{'ingredient':24s} {'pd':>6s} {'delta pd [95%]':>26s}   {'worst (held-out)':>16s} "
        f"{'delta worst [95%]':>26s}   {'pd on baseline worst':>20s} {'delta [95%]':>26s}"
    ]
    for scored, deltas, on_target, target_delta in rows:
        by = {d.metric: d for d in deltas}
        lines.append(
            f"{scored.label:24s} {scored.score.pd:6.3f} {_ci(by['pd_at_operating_point']):>26s}   "
            f"{scored.score.worst_tactic_pd:16.3f} {_ci(by[METRIC_WORST]):>26s}   "
            f"{on_target:20.3f} {_ci(target_delta):>26s}"
        )
    return "\n".join(lines)


def format_policy(rows: Sequence[tuple[str, Scored, Scored, list[PairedDelta]]]) -> str:
    lines = [
        f"{'fleet':24s} {'drones':>6s} {'$/h':>5s} {'tactics':>7s}   {'naive worst [95%]':>24s}   "
        f"{'policy worst [95%]':>24s}   {'delta worst [95%]':>26s}   {'naive pd':>8s} "
        f"{'policy pd':>9s}"
    ]
    for name, naive, fixed, deltas in rows:
        worst = next(d for d in deltas if d.metric == METRIC_WORST)
        n, f = naive.score, fixed.score
        n_ci, f_ci = n.worst_tactic_pd_ci, f.worst_tactic_pd_ci
        lines.append(
            f"{name:24s} {sum(a.type == 'drone' for a in naive.fleet.agents):6d} "
            f"{naive.fleet.cost_per_hour():5.0f} {len(n.pd_by_tactic):7d}   "
            f"{n.worst_tactic_pd:6.3f} [{n_ci[0]:.3f}, {n_ci[1]:.3f}]   "
            f"{f.worst_tactic_pd:6.3f} [{f_ci[0]:.3f}, {f_ci[1]:.3f}]   "
            f"{_ci(worst):>26s}   {n.pd:8.3f} {f.pd:9.3f}"
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
    parser.add_argument("--held-out", action="store_true", help="confirm on unseen seeds")
    parser.add_argument("--held-out-seeds", type=int, default=None, help="default: all that exist")
    parser.add_argument("--policy-fleets", nargs="*", default=[], help="apply the policy to these")
    parser.add_argument("--export-pairs", action="store_true", help="(re)export replay pairs")
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

    held = None
    if args.held_out:
        t3 = time.perf_counter()
        ho_seeds, ho_quiet = held_out_split(
            args.seeds_file, len(seeds), len(quiet_seeds), len(s_quiet), args.held_out_seeds
        )
        held = held_out_confirmation(
            inputs,
            best.candidate,
            conf,
            ho_seeds,
            ho_quiet,
            [*s_seeds, *s_quiet],
            args.sweep_dir,
            args.workers,
        )
        print(
            f"held-out confirmation: {len(ho_seeds)} seeds, {len(ho_quiet)} quiet nights, "
            f"{len(held.tactics)} tactics, {time.perf_counter() - t3:.0f} s -> {held.verdict.upper()}"
        )
        for d in held.deltas:
            print(f"  {d.metric:26s} {_ci(d)}")
        t4 = time.perf_counter()
        parts = ingredients(inputs, best.candidate, held, args.sweep_dir, args.workers)
        fix = fix.model_copy(update={"ingredients": ingredient_rows(parts)})
        print(f"ingredients: {time.perf_counter() - t4:.0f} s")
        print(format_ingredients(parts))
    if args.policy_fleets:
        t5 = time.perf_counter()
        table = policy_by_fleet(
            inputs,
            best.candidate,
            [inputs.baseline, *args.policy_fleets],
            seeds,
            quiet_seeds,
            args.sweep_dir,
            args.workers,
            fixed_name=fixed_name,
        )
        fix = fix.model_copy(update={"policy_by_fleet": policy_rows(table)})
        print(f"policy by fleet: {time.perf_counter() - t5:.0f} s")
        print(format_policy(table))

    report, detail_fix = add_fix_to_report(
        report, detail, inputs, best.candidate, fixed_name, fix, held
    )
    print("written:", *write_report(report, detail_fix, args.out))
    swept = dataclasses.replace(result, inputs=dataclasses.replace(inputs, tactics=conf.tactics))
    write_tactics(swept, args.out / "report_tactics")
    print("fixed fleet:", *write_fixed_fleet(best.candidate, fixed_name, args.out / "fixed"))
    if args.export_pairs and conf.is_fix:
        pairs, note = export_pairs(inputs, best.candidate, conf, args.out / "replays" / "pairs")
        print(f"replay pairs: {len(pairs)} for {note}")
    print(f"total wall time {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
