"""The overnight campaign: the cheapest defensible configurations, confirmed on unseen seeds.

    python -m airtight.score.campaign --deadline-hours 6 --out /abs/path/data/campaign
    python -m airtight.score.campaign --smoke --out /abs/path/data/campaign

What is optimised is only what the defender controls: the patrol policy (policy.py), the charge
schedule, dock assignment, fleet size, and purchasable hardware (hardware.py). The site's
geometry, the sensor curve, the intruder, the speed limits, the response time and the benign
traffic are never touched in a headline row. Task-time and response-time what-ifs live in
their own stages and every row they produce is marked strict = False.

Seeds (seedsplit.py): search seeds pick policies, validation seeds pick one policy per
hardware configuration, the finalists, the recommended configuration, the random tactics that
hurt most and each configuration's worst tactic. Final seeds only report. The evaluator
refuses a seed outside the role a stage declares.

Worst-tactic detection is reported two ways. "naive" is the minimum over tactics on the final
seeds: selecting the minimum of many noisy estimates reads low. "held-out" takes the tactic
that was worst on the VALIDATION seeds and scores that one tactic on the final seeds: unbiased
for that tactic, so it reads at or above the true minimum. The truth lies between them.

Stages, in priority order, each sized to its share of --deadline-hours from a measured rate,
each checkpointed, each resumable from the cache: probe, A (baseline and the confirmed fix
against the strong adversary), search (successive halving per hardware configuration),
validation, final_standin (every configuration, the frontier), assumption60, final_strong
(the finalists), sensitivity, audit. Past the deadline no new stage starts. Reporting is a
separate step (campaign_report) that reads results.json and the cache.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import airtight
from airtight.contracts import FleetConfig, SensorCurves, Site
from airtight.score import adversary, hardware, roc, seedsplit
from airtight.score.cells import (
    QUIET_KEY,
    CacheFull,
    Config,
    Evaluator,
    Request,
    episode_task,
    quiet_cost_s,
    quiet_task,
    tactic_cost_s,
)
from airtight.score.policy import Policy, named_policy, perturb, sample_policy
from airtight.score.sweep import PHASE_DEDUP, gap_phases, grid_tactics
from airtight.sim.constants import ENGINE_IGNORES, ENGINE_VERSION, TAU_INVESTIGATE
from airtight.sim.coverage import uncovered_intervals
from airtight.sim.episode import PARAMS_JSON_ENV, TASK_TIME_ENV, official_params

if TYPE_CHECKING:
    from collections.abc import Sequence

    from airtight.contracts import Tactic
    from airtight.score.cells import Array

FAR_TARGET = 1.0
TARGET = 0.95
KNEE = 0.02  # recommended = the cheapest configuration within this of the best objective
SHARES = {
    "A": 0.10,
    "search": 0.31,
    "validation": 0.13,
    "final_standin": 0.14,
    "final_strong": 0.12,
    "assumption60": 0.06,
    "sensitivity": 0.06,
    "final_strong_nocams": 0.05,
    "audit": 0.03,
}
STAGES = ("probe", *SHARES)
SLACK = (
    0.75  # the share of a stage's time its plan may fill; the rest absorbs what the probe missed
)
ASSUMED_TASK_TIME_S = 60.0
SENS_TASK_TIMES_S = (0.0, 30.0, 60.0, 120.0)
SENS_RESPONSE_TIMES_S = (10.0, 20.0, 30.0)
POLICY_STREAM = 78
RANDOM_TACTIC_KEY = (20260919, 79)
FRESH_TACTIC_KEY = (20260919, 80)
AUDIT_KEY = (20260919, 81)
BASELINE_LABEL = "baseline"
FIX_LABEL = "confirmed_fix"
BEST_FREE_LABEL = "best_free_policy"


@dataclass(frozen=True)
class Sizes:
    """Every size a stage may use. Budgets pick inside [min, max]."""

    specs: tuple[str, ...] | None  # hardware names to keep; None keeps all
    n0: tuple[int, int]  # random policies per hardware configuration at rung 0
    rung_seeds: tuple[int, int, int]
    rung_quiet: tuple[int, int, int]
    n_perturb: int
    n_validate: int  # survivors per hardware configuration that go to validation
    val_seeds: tuple[int, int]
    val_quiet: int
    final_seeds: tuple[int, int]
    final_quiet: int
    strong_phases: int
    n_random: int
    keep_random: int
    screen_seeds: int
    strong_val_seeds: tuple[int, int]
    strong_final_seeds: tuple[int, int]
    n_worst: int
    worst_seeds: int
    strong_quiet: int
    max_picks: int
    sens_seeds: tuple[int, int]
    n_audit_rows: int
    n_fresh: int
    fresh_seeds: int
    n_boot: int
    n_named: int = 5  # named policies that seed every search
    reduced_phases: tuple[int, int] = (4, 2)  # at the speed cap, at the minimum speed
    standin_phases: tuple[int, int] = (16, 8)
    max_lane_c: int | None = None
    sens_task_times_s: tuple[float, ...] = SENS_TASK_TIMES_S
    sens_response_times_s: tuple[float, ...] = SENS_RESPONSE_TIMES_S


FULL = Sizes(
    specs=None,
    n0=(10, 120),
    rung_seeds=(4, 12, 40),
    rung_quiet=(2, 5, 12),
    n_perturb=4,
    n_validate=2,
    val_seeds=(20, 200),
    val_quiet=16,
    final_seeds=(40, 600),
    final_quiet=40,
    strong_phases=48,
    n_random=300,
    keep_random=30,
    screen_seeds=10,
    strong_val_seeds=(10, 40),
    strong_final_seeds=(40, 400),
    n_worst=5,
    worst_seeds=1200,
    strong_quiet=140,
    max_picks=2,
    sens_seeds=(20, 200),
    n_audit_rows=200,
    n_fresh=300,
    fresh_seeds=10,
    n_boot=500,
)
SMOKE = Sizes(
    specs=("d2_std_nocams", "d1_std_nocams", "d3_swap_cams"),
    n0=(2, 2),
    rung_seeds=(2, 3, 4),
    rung_quiet=(1, 1, 2),
    n_perturb=1,
    n_validate=1,
    val_seeds=(3, 3),
    val_quiet=2,
    final_seeds=(4, 4),
    final_quiet=2,
    strong_phases=2,
    n_random=8,
    keep_random=2,
    screen_seeds=2,
    strong_val_seeds=(2, 2),
    strong_final_seeds=(3, 3),
    n_worst=2,
    worst_seeds=6,
    strong_quiet=2,
    max_picks=1,
    sens_seeds=(2, 2),
    n_audit_rows=10,
    n_fresh=6,
    fresh_seeds=2,
    n_boot=50,
    n_named=2,
    reduced_phases=(2, 1),
    standin_phases=(2, 1),
    max_lane_c=3,
    sens_task_times_s=(0.0, 60.0),
    sens_response_times_s=(20.0, 30.0),
)


@dataclass
class Entry:
    """One configuration the campaign knows by label."""

    label: str
    hardware: str
    cost_per_hour: float
    config: Config
    policy: dict[str, object] | None  # Policy.describe(); None means the scenario's own file


@dataclass
class Ctx:
    out: Path
    sizes: Sizes
    site: Site
    curves: SensorCurves
    baseline: FleetConfig
    limits: adversary.Limits
    costs: dict[str, float]
    seeds: seedsplit.SeedSplit
    ev: Evaluator
    lane_c_tactics: list[Tactic]
    started: float
    deadline: float
    checkpoint: dict[str, Any]
    tactics_seen: dict[str, Tactic] = field(default_factory=dict)
    configs_seen: dict[str, Config] = field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return self.deadline - self.started

    def say(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {text}", flush=True)

    def budget_sim_s(self, stage: str) -> float:
        """Simulated seconds a stage may plan for."""
        return SHARES[stage] * self.total_s * float(self.checkpoint["rate"]) * SLACK

    def stage_stop(self, stage: str) -> float:
        """A stage is cut at 1.5 times its share, or at the deadline, whichever is first."""
        return min(self.deadline, time.time() + 1.5 * SHARES[stage] * self.total_s)

    def save_checkpoint(self) -> None:
        _write_json(self.out / "checkpoint.json", self.checkpoint)

    def run(self, requests: Sequence[Request], role: str, stop_at: float | None = None) -> int:
        for request in requests:
            self.configs_seen[request.config.key(self.curves)] = request.config
            for tactic in request.tactics:
                self.tactics_seen[tactic.content_hash()] = tactic
        return self.ev.run(requests, role, stop_at)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(path)


def _fit(budget: float, per_unit: float, fixed: float, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    if per_unit <= 0:
        return hi
    return int(min(max((budget - fixed) // per_unit, lo), hi))


# ---- tactic sets -----------------------------------------------------------------------------


def gap_tactics(site: Site, fleets: Sequence[FleetConfig], speed_mps: float) -> list[Tactic]:
    """The midpoint of every gap of these fleets (nobody on duty, no drone on duty), from every
    entry, straight to the asset. The adversary knows the schedule, so a configuration's own
    gaps are always attacked."""
    template = grid_tactics(site, speed_mps, 1)[0]
    return [
        template.model_copy(
            update={"id": f"gap-{entry.id}-{phase:.4f}", "entry_id": entry.id, "phase": phase}
        )
        for entry in site.entry_points
        for phase in gap_phases(list(fleets), [])
    ]


def _slow(tactics: Sequence[Tactic], speed_mps: float) -> list[Tactic]:
    return [t.model_copy(update={"id": f"slow-{t.id}", "speed_mps": speed_mps}) for t in tactics]


def reduced_common(
    site: Site, limits: adversary.Limits, phases: tuple[int, int] = (4, 2)
) -> list[Tactic]:
    """The search's small set: every entry at 4 phases at the speed cap, 2 at the minimum."""
    fast = grid_tactics(site, limits.speed_cap_mps, phases[0])
    slow = [
        t.model_copy(update={"phase": t.phase + 0.125}) for t in grid_tactics(site, 1.0, phases[1])
    ]
    return [*fast, *_slow(slow, limits.speed_min_mps)]


def standin_common(
    site: Site, limits: adversary.Limits, phases: tuple[int, int] = (16, 8)
) -> list[Tactic]:
    """The stand-in adversary: the sweep's 16-phase grid at the cap, plus 8 phases at the
    minimum speed."""
    return [
        *grid_tactics(site, limits.speed_cap_mps, phases[0]),
        *_slow(grid_tactics(site, 1.0, phases[1]), limits.speed_min_mps),
    ]


def with_gaps(
    common: Sequence[Tactic], site: Site, fleets: Sequence[FleetConfig], speed_mps: float
) -> list[Tactic]:
    """common plus the gap tactics of these fleets, skipping a gap tactic that repeats an
    attack already present (same entry and speed, phase within PHASE_DEDUP)."""
    out = list(common)
    have = [
        (t.entry_id, t.phase) for t in out if t.speed_mps == speed_mps and len(t.waypoints) == 1
    ]
    for tactic in gap_tactics(site, fleets, speed_mps):
        if any(e == tactic.entry_id and abs(p - tactic.phase) < PHASE_DEDUP for e, p in have):
            continue
        have.append((tactic.entry_id, tactic.phase))
        out.append(tactic)
    return out


# ---- scoring ---------------------------------------------------------------------------------


def _ci(values: Array) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


@dataclass
class Scored:
    summary: dict[str, Any]
    reps: roc.Replicates
    tactic_ids: list[str]


def score(
    ctx: Ctx,
    entry: Entry,
    tactics: Sequence[Tactic],
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    role: str,
    adversary_name: str,
    config: Config | None = None,
    assumption: dict[str, float] | None = None,
    heldout_worst: tuple[Tactic, Sequence[int]] | None = None,
    n_boot: int | None = None,
    quiet_config: Config | None = None,
) -> Scored:
    """Score cached results. config, when given, is the variant actually simulated (a task
    time or a response time what-if). Quiet nights come from quiet_config, by default
    entry.config: a quiet night never reads the task time, so a task-time variant shares the
    strict configuration's nights. It DOES read the response time (the band ring of the
    patrol weight is v_ref * response_time_s), so a response-time variant passes its own."""
    sim_config = config or entry.config
    peaks = ctx.ev.peaks(sim_config, tactics, seeds)
    quiet = ctx.ev.quiet(quiet_config or entry.config, quiet_seeds)
    n_boot = ctx.sizes.n_boot if n_boot is None else n_boot
    op = roc.operating_point(peaks, quiet, FAR_TARGET)
    per_tactic, worst = roc.pd_by_tactic(peaks, op.tau)
    reps = roc.replicates(peaks, quiet, n_boot, 0, FAR_TARGET)
    ids = [t.id for t in tactics]
    summary: dict[str, Any] = {
        "label": entry.label,
        "config": sim_config.name,
        "hardware": entry.hardware,
        "cost_per_hour": entry.cost_per_hour,
        "adversary": adversary_name,
        "seed_role": role,
        "n_seeds": len(seeds),
        "n_quiet": len(quiet_seeds),
        "n_tactics": len(tactics),
        "strict": assumption is None,
        "assumption": assumption or {},
        "tau": op.tau,
        "flag": op.flag,
        "tau_on_floor": bool(op.tau <= TAU_INVESTIGATE),
        "far_per_hour": op.far,
        "pd": op.pd,
        "pd_ci": _ci(reps.pd_by_tactic.mean(axis=1)),
        "worst_naive": {
            "tactic": ids[worst],
            "pd": float(per_tactic[worst]),
            "ci": _ci(reps.pd_by_tactic[:, worst]),
        },
        "objective": 0.5 * op.pd + 0.5 * float(per_tactic[worst]),
        "pd_by_tactic": {i: float(v) for i, v in zip(ids, per_tactic, strict=True)},
    }
    if heldout_worst is not None:
        tactic, more_seeds = heldout_worst
        column = ctx.ev.peaks(sim_config, [tactic], more_seeds)
        col_reps = roc.replicates(column, quiet, n_boot, 0, FAR_TARGET)
        col_op = roc.operating_point(column, quiet, FAR_TARGET)
        # the point estimate and its interval both use the column's own operating point
        per_column, _ = roc.pd_by_tactic(column, col_op.tau)
        detected = float(per_column[0])
        summary["worst_heldout"] = {
            "tactic": tactic.id,
            "pd": detected,
            "ci": _ci(col_reps.pd_by_tactic[:, 0]),
            "n_seeds": len(more_seeds),
            "tau_single_column": col_op.tau,
        }
    return Scored(summary, reps, ids)


def paired(a: Scored, b: Scored) -> dict[str, Any]:
    """a minus b with paired bootstrap intervals. Needs the same seeds and quiet counts."""
    if not (
        np.array_equal(a.reps.row_w, b.reps.row_w)
        and np.array_equal(a.reps.quiet_w, b.reps.quiet_w)
    ):
        raise ValueError("the two configurations were not resampled with the same indices")
    wa = a.tactic_ids.index(a.summary["worst_naive"]["tactic"])
    wb = b.tactic_ids.index(b.summary["worst_naive"]["tactic"])
    return {
        "a": a.summary["label"],
        "b": b.summary["label"],
        "n_seeds": a.summary["n_seeds"],
        "pd_delta": a.summary["pd"] - b.summary["pd"],
        "pd_delta_ci": _ci(a.reps.pd_by_tactic.mean(axis=1) - b.reps.pd_by_tactic.mean(axis=1)),
        "worst_delta": a.summary["worst_naive"]["pd"] - b.summary["worst_naive"]["pd"],
        "worst_delta_ci": _ci(a.reps.pd_by_tactic[:, wa] - b.reps.pd_by_tactic[:, wb]),
        "worst_delta_kind": "naive: each side's minimum on these seeds, tactics held fixed",
    }


# ---- entries ---------------------------------------------------------------------------------


def baseline_entry(ctx: Ctx) -> Entry:
    config = Config(ctx.baseline.name, ctx.site, ctx.baseline, official_params())
    return Entry(BASELINE_LABEL, "d2_std_nocams", ctx.baseline.cost_per_hour(), config, None)


def fix_entry(ctx: Ctx) -> Entry:
    """The confirmed fix of step F2: uniform weight, the Go2 half a cycle out of phase."""
    policy = named_policy(ctx.baseline, "uniform", 0.0, True)
    fleet = policy.apply(ctx.baseline, f"{ctx.baseline.name}_fixed")
    params = dataclasses.replace(official_params(), weight_mode="uniform")
    config = Config(fleet.name, ctx.site, fleet, params)
    return Entry(FIX_LABEL, "d2_std_nocams", fleet.cost_per_hour(), config, policy.describe())


def specs_for(ctx: Ctx) -> list[hardware.HardwareSpec]:
    keep = ctx.sizes.specs
    return [s for s in hardware.all_specs() if keep is None or s.name in keep]


def policy_entry(ctx: Ctx, spec: hardware.HardwareSpec, policy: Policy, label: str) -> Entry:
    site = hardware.build_site(ctx.site, ctx.curves, spec)
    fleet = policy.apply(hardware.build_fleet(ctx.baseline, spec), None)
    fleet = fleet.model_copy(update={"name": f"{spec.name}__p{policy.digest()}"})
    cost = hardware.cost_per_hour(fleet, site, ctx.site, spec, ctx.costs)
    config = Config(fleet.name, site, fleet, policy.params())
    return Entry(label, spec.name, cost, config, policy.describe())


def _policy_from(described: dict[str, Any]) -> Policy:
    fields = dict(described)
    fields["offsets"] = tuple(sorted((str(a), float(v)) for a, v in fields["offsets"].items()))
    fields["docks"] = tuple(sorted((str(a), str(d)) for a, d in fields["docks"].items()))
    return Policy(**fields)


def _spec_by_name(name: str) -> hardware.HardwareSpec:
    return next(s for s in hardware.all_specs() if s.name == name)


def entry_from_record(ctx: Ctx, record: dict[str, Any]) -> Entry:
    if record["label"] == BASELINE_LABEL:
        return baseline_entry(ctx)
    if record["label"] == FIX_LABEL:
        return fix_entry(ctx)
    policy = _policy_from(record["policy"])
    return policy_entry(ctx, _spec_by_name(record["hardware"]), policy, record["label"])


def record_of(entry: Entry) -> dict[str, Any]:
    return {
        "label": entry.label,
        "hardware": entry.hardware,
        "cost_per_hour": entry.cost_per_hour,
        "policy": entry.policy,
        "config": entry.config.name,
        "fleet_hash": entry.config.fleet.content_hash(),
        "site_hash": entry.config.site.content_hash(),
        "duty": duty_shares(entry.config.fleet),
    }


def duty_shares(fleet: FleetConfig) -> dict[str, dict[str, float]]:
    """For every agent that charges: its on-duty share inside the window an attack can fall in
    ([0, reference cycle), which is what Tactic.phase spans) next to its steady-state share
    endurance / cycle. A large difference means the window flatters or punishes the agent: a
    charge period pushed outside the window is never seen by any tactic or quiet night."""
    out = {}
    for agent in fleet.agents:
        if agent.charge_time_s <= 0:
            continue
        down = sum(g.end_phase - g.start_phase for g in uncovered_intervals(fleet, only=[agent.id]))
        out[agent.id] = {
            "in_window": 1.0 - float(down),
            "steady_state": float(agent.endurance_s / (agent.endurance_s + agent.charge_time_s)),
        }
    return out


# ---- stages ----------------------------------------------------------------------------------


def stage_probe(ctx: Ctx) -> dict[str, Any]:
    """Measure simulated seconds per wall second through the real pool."""
    entry = baseline_entry(ctx)
    tactics = with_gaps(
        reduced_common(ctx.site, ctx.limits, ctx.sizes.reduced_phases),
        ctx.site,
        [ctx.baseline],
        2.0,
    )
    n = ctx.sizes.rung_seeds[1]
    before, start = ctx.ev.sim_seconds, time.time()
    ctx.run(
        [Request(entry.config, tuple(tactics), ctx.seeds.intrusion["search"][:n])],
        "search",
    )
    wall = time.time() - start
    simulated = ctx.ev.sim_seconds - before
    known = ctx.checkpoint.get("rate")
    rate = simulated / wall if simulated > 0 and wall > 0 else known
    if rate is None:
        raise RuntimeError("the probe simulated nothing and no earlier rate is on record")
    ctx.checkpoint["rate"] = float(rate)
    per_episode = float(np.mean([tactic_cost_s(ctx.site, t, entry.config.params) for t in tactics]))
    ctx.say(
        f"probe: {rate:.0f} simulated s per wall s on {ctx.ev.workers} workers, "
        f"about {rate / per_episode:.1f} episodes per second"
    )
    return {"rate_sim_s_per_wall_s": float(rate), "episodes_per_s": rate / per_episode}


def strong_tactics(
    ctx: Ctx, entries: Sequence[Entry], stage: str
) -> tuple[list[Tactic], dict[str, list[str]]]:
    """The strong adversary for these configurations: the 48-phase grid at both speed limits,
    lane C's real tactics, every configuration's gap tactics, and for each configuration the
    random multi-waypoint tactics that hurt it most on validation seeds. The union is used
    for everyone, so every comparison stays paired."""
    sizes = ctx.sizes
    base = [
        *adversary.strong_grid(ctx.site, ctx.limits, sizes.strong_phases),
        *ctx.lane_c_tactics,
    ]
    base = with_gaps(base, ctx.site, [e.config.fleet for e in entries], ctx.limits.speed_cap_mps)
    randoms = adversary.random_tactics(ctx.site, ctx.limits, sizes.n_random, RANDOM_TACTIC_KEY)
    seeds = ctx.seeds.intrusion["validation"][: sizes.screen_seeds]
    quiet = ctx.seeds.quiet["validation"][: sizes.val_quiet]
    ctx.run([Request(e.config, tuple(randoms), seeds, quiet) for e in entries], "validation")
    kept: dict[str, list[str]] = {}
    union: dict[str, Tactic] = {}
    for entry in entries:
        scored = score(ctx, entry, randoms, seeds, quiet, "validation", "random screen", n_boot=2)
        harmful = adversary.keep_most_harmful(
            randoms, scored.summary["pd_by_tactic"], sizes.keep_random
        )
        kept[entry.label] = [t.id for t in harmful]
        union.update({t.id: t for t in harmful})
    ctx.say(f"{stage}: {len(union)} random tactics kept across {len(entries)} configurations")
    return [*base, *(union[i] for i in sorted(union))], kept


def strong_evaluate(ctx: Ctx, entries: Sequence[Entry], stage: str) -> dict[str, Any]:
    sizes = ctx.sizes
    tactics, kept = strong_tactics(ctx, entries, stage)
    quiet_val = ctx.seeds.quiet["validation"][: sizes.val_quiet]
    quiet_final = ctx.seeds.quiet["final"][: sizes.strong_quiet]
    per_seed = sum(
        tactic_cost_s(e.config.site, t, e.config.params) for e in entries for t in tactics
    )
    fixed = sum(quiet_cost_s(e.config.fleet, e.config.params) for e in entries) * len(quiet_final)
    budget = ctx.budget_sim_s(stage)
    n_val = _fit(0.15 * budget, per_seed, 0.0, sizes.strong_val_seeds)
    n_final = _fit(0.85 * budget, per_seed, fixed, sizes.strong_final_seeds)
    plan = ctx.checkpoint.setdefault("plans", {}).setdefault(
        stage, {"n_val": n_val, "n_final": n_final}
    )
    ctx.save_checkpoint()
    n_val, n_final = int(plan["n_val"]), int(plan["n_final"])
    ctx.say(f"{stage}: {len(tactics)} tactics; {n_val} validation and {n_final} final seeds")
    stop = ctx.stage_stop(stage)

    val_seeds = ctx.seeds.intrusion["validation"][:n_val]
    done = ctx.run(
        [Request(e.config, tuple(tactics), val_seeds, quiet_val) for e in entries],
        "validation",
        stop,
    )
    val_seeds = val_seeds[:done]
    final_seeds = ctx.seeds.intrusion["final"][:n_final]
    done = ctx.run(
        [Request(e.config, tuple(tactics), final_seeds, quiet_final) for e in entries],
        "final",
        stop,
    )
    final_seeds = final_seeds[:done]
    if not val_seeds or not final_seeds:
        return {"complete": False, "reason": "cut before any seed finished", "kept_random": kept}

    worst_by_label: dict[str, list[Tactic]] = {}
    validation_rows = []
    for entry in entries:
        val = score(ctx, entry, tactics, val_seeds, quiet_val, "validation", "strong")
        order = sorted(
            val.summary["pd_by_tactic"], key=lambda i: (val.summary["pd_by_tactic"][i], i)
        )
        by_id = {t.id: t for t in tactics}
        worst_by_label[entry.label] = [by_id[i] for i in order[: sizes.n_worst]]
        validation_rows.append(val.summary)
    more = ctx.seeds.intrusion["final"][: sizes.worst_seeds]
    done = ctx.run(
        [Request(e.config, tuple(worst_by_label[e.label]), more) for e in entries], "final", stop
    )
    more = more[: max(done, 0)]
    rows, scored_by_label = [], {}
    for entry in entries:
        worst = worst_by_label[entry.label][0]
        held = (worst, more if len(more) >= len(final_seeds) else final_seeds)
        scored = score(
            ctx, entry, tactics, final_seeds, quiet_final, "final", "strong", heldout_worst=held
        )
        scored.summary["validation_worst_tactics"] = [t.id for t in worst_by_label[entry.label]]
        if len(more) >= len(final_seeds):
            five = ctx.ev.peaks(entry.config, worst_by_label[entry.label], more)
            at_tau, _ = roc.pd_by_tactic(five, scored.summary["tau"])
            scored.summary["validation_worst_on_final"] = {
                t.id: float(v) for t, v in zip(worst_by_label[entry.label], at_tau, strict=True)
            }
            scored.summary["validation_worst_on_final_n_seeds"] = len(more)
        # the quoted false alarm rate is in-sample: the threshold was picked on the nights it
        # is measured on. Pick it on one half of the nights and measure it on the other.
        nights = ctx.ev.quiet(entry.config, quiet_final)
        half = len(nights) // 2
        if half >= 1:
            all_peaks = ctx.ev.peaks(entry.config, tactics, final_seeds)
            tau_half = roc.operating_point(all_peaks, nights[:half], FAR_TARGET).tau
            scored.summary["far_check"] = {
                "tau_from_first_half": tau_half,
                "far_on_second_half": float(roc.far_curve(nights[half:], np.array([tau_half]))[0]),
                "nights_each": [half, len(nights) - half],
            }
        rows.append(scored.summary)
        scored_by_label[entry.label] = scored
    deltas = [
        paired(scored_by_label[e.label], scored_by_label[BASELINE_LABEL])
        for e in entries
        if e.label != BASELINE_LABEL and BASELINE_LABEL in scored_by_label
    ]
    return {
        "complete": True,
        "entries": [record_of(e) for e in entries],
        "validation": validation_rows,
        "final": rows,
        "paired_vs_baseline": deltas,
        "kept_random": kept,
        "n_lane_c_tactics": len(ctx.lane_c_tactics),
    }


def stage_a(ctx: Ctx) -> dict[str, Any]:
    return strong_evaluate(ctx, [baseline_entry(ctx), fix_entry(ctx)], "A")


def _rank(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (-r["objective"], -r["pd"], r["digest"]))


NAMED_SEEDS = (
    ("asset", 0.0, False),
    ("uniform", 0.0, True),
    ("asset", 1.0, False),
    ("uniform", 1.0, True),
    ("band", 1.0, True),
)


def _named_seeds(fleet: FleetConfig, n: int) -> list[Policy]:
    """The baseline's policy, the confirmed fix, and three staggered variants, de-duplicated
    (a fleet with no Go2 or one drone makes some of them coincide)."""
    found: dict[str, Policy] = {}
    for mode, fraction, half in NAMED_SEEDS[:n]:
        policy = named_policy(fleet, mode, fraction, half)
        found.setdefault(policy.digest(), policy)
    return list(found.values())


def stage_search(ctx: Ctx) -> dict[str, Any]:
    """Successive halving for every hardware configuration, rung by rung across all of them,
    so a cut hits every configuration alike."""
    sizes, specs = ctx.sizes, specs_for(ctx)
    common = reduced_common(ctx.site, ctx.limits, ctx.sizes.reduced_phases)
    cap = ctx.limits.speed_cap_mps
    stop = ctx.stage_stop("search")

    def requests_for(
        pool: dict[str, list[Policy]], n_seeds: int, n_quiet: int
    ) -> list[tuple[str, Policy, Entry, list[Tactic], Request]]:
        out = []
        for spec in specs:
            for policy in pool[spec.name]:
                entry = policy_entry(ctx, spec, policy, f"{spec.name}:{policy.digest()}")
                tactics = with_gaps(common, entry.config.site, [entry.config.fleet], cap)
                request = Request(
                    entry.config,
                    tuple(tactics),
                    ctx.seeds.intrusion["search"][:n_seeds],
                    ctx.seeds.quiet["search"][:n_quiet],
                )
                out.append((spec.name, policy, entry, tactics, request))
        return out

    def evaluate(
        pool: dict[str, list[Policy]], rung: int, tag: str
    ) -> dict[str, list[dict[str, Any]]]:
        n_seeds, n_quiet = sizes.rung_seeds[rung], sizes.rung_quiet[rung]
        items = requests_for(pool, n_seeds, n_quiet)
        done = ctx.run([item[4] for item in items], "search", stop)
        ranked: dict[str, list[dict[str, Any]]] = {s.name: [] for s in specs}
        if done == 0:
            return ranked
        for spec_name, policy, entry, tactics, request in items:
            s = score(
                ctx,
                entry,
                tactics,
                request.seeds[:done],
                request.quiet_seeds,
                "search",
                "reduced",
                n_boot=2,
            ).summary
            ranked[spec_name].append(
                {
                    "digest": policy.digest(),
                    "origin": policy.origin,
                    "policy": policy.describe(),
                    "objective": s["objective"],
                    "pd": s["pd"],
                    "worst": s["worst_naive"]["pd"],
                    "worst_tactic": s["worst_naive"]["tactic"],
                    "tau": s["tau"],
                    "flag": s["flag"],
                    "n_seeds": done,
                    "rung": tag,
                }
            )
            ctx.ev.forget(entry.config)
        return {name: _rank(rows) for name, rows in ranked.items()}

    # size rung 0 from the budget: a candidate's whole path through the rungs, amortised
    probe_entry = policy_entry(ctx, specs[0], _named_seeds(ctx.baseline, 1)[0], "size")
    ep = float(np.mean([tactic_cost_s(ctx.site, t, probe_entry.config.params) for t in common]))
    ep *= len(common) + 4
    quiet = quiet_cost_s(ctx.baseline, probe_entry.config.params)
    s0, s1, s2 = sizes.rung_seeds
    q0, q1, q2 = sizes.rung_quiet
    rung0 = ep * s0 + quiet * q0
    rung1 = ep * s1 + quiet * q1
    rung2 = ep * s2 + quiet * q2
    per_candidate = rung0 + rung1 / 5 + rung2 / 25
    fixed = 2 * sizes.n_perturb * rung1 + 2 * rung2 + sizes.n_named * per_candidate
    n0 = _fit(ctx.budget_sim_s("search") / len(specs), per_candidate, fixed, sizes.n0)
    plan = ctx.checkpoint.setdefault("plans", {}).setdefault("search", {"n0": n0})
    ctx.save_checkpoint()
    n0 = int(plan["n0"])
    ctx.say(f"search: {len(specs)} hardware configurations, {n0} random policies each at rung 0")

    pool: dict[str, list[Policy]] = {}
    for index, spec in enumerate(specs):
        fleet = hardware.build_fleet(ctx.baseline, spec)
        site = hardware.build_site(ctx.site, ctx.curves, spec)
        rng = np.random.default_rng([20260919, POLICY_STREAM, index, 0])
        named = _named_seeds(fleet, sizes.n_named)
        pool[spec.name] = [*named, *(sample_policy(site, fleet, rng) for _ in range(n0))]
    by_digest = {s.name: {p.digest(): p for p in pool[s.name]} for s in specs}
    history: dict[str, list[dict[str, Any]]] = {s.name: [] for s in specs}
    survivors = pool
    for rung in range(3):
        ranked = evaluate(survivors, rung, f"rung{rung}")
        if not any(ranked.values()):
            ctx.say(f"search: rung {rung} was cut before any seed finished")
            break
        # from the first configuration's pool, as launched: a rescore must retrace the same
        # survivors, or it would cascade into new finalists and hours of new episodes
        keep = max(2, -(-len(next(iter(survivors.values()))) // 5))
        survivors = {}
        for spec in specs:
            history[spec.name] += ranked[spec.name]
            survivors[spec.name] = [
                by_digest[spec.name][r["digest"]] for r in ranked[spec.name][:keep]
            ]
        ctx.say(f"search: rung {rung} done, {keep} survivors per configuration")

    # local perturbation round the two best, then the best of those at the top rung
    near: dict[str, list[Policy]] = {}
    for index, spec in enumerate(specs):
        fleet = hardware.build_fleet(ctx.baseline, spec)
        site = hardware.build_site(ctx.site, ctx.curves, spec)
        rng = np.random.default_rng([20260919, POLICY_STREAM, index, 1])
        near[spec.name] = [
            perturb(site, fleet, parent, rng)
            for parent in survivors[spec.name][:2]
            for _ in range(sizes.n_perturb)
        ]
        by_digest[spec.name].update({p.digest(): p for p in near[spec.name]})
    ranked = evaluate(near, 1, "near1")
    top_near = {s.name: [by_digest[s.name][r["digest"]] for r in ranked[s.name][:2]] for s in specs}
    ranked2 = evaluate(top_near, 2, "near2") if any(ranked.values()) else {}
    result: dict[str, Any] = {"n0": n0, "specs": {}}
    for spec in specs:
        history[spec.name] += ranked.get(spec.name, []) + ranked2.get(spec.name, [])
        top_rung = [r for r in history[spec.name] if r["rung"] in ("rung2", "near2")]
        best = _rank(top_rung or history[spec.name])
        chosen, seen = [], set()
        for row in best:
            if row["digest"] not in seen:
                seen.add(row["digest"])
                chosen.append(row)
        result["specs"][spec.name] = {
            "survivors": chosen[: sizes.n_validate],
            "history": history[spec.name],
        }
    return result


def _search_survivors(ctx: Ctx, search: dict[str, Any]) -> dict[str, list[Policy]]:
    out = {}
    for spec in specs_for(ctx):
        rows = search["specs"][spec.name]["survivors"]
        out[spec.name] = [_policy_from(r["policy"]) for r in rows]
    return out


def stage_validation(ctx: Ctx, search: dict[str, Any]) -> dict[str, Any]:
    """Survivors on validation seeds against the full stand-in set. Picks one policy per
    hardware configuration, then the finalists and the recommended configuration, all from
    validation numbers, so the final seeds stay unseen."""
    sizes, specs = ctx.sizes, specs_for(ctx)
    common = standin_common(ctx.site, ctx.limits, ctx.sizes.standin_phases)
    cap = ctx.limits.speed_cap_mps
    survivors = _search_survivors(ctx, search)
    items = []
    for spec in specs:
        for policy in survivors[spec.name]:
            entry = policy_entry(ctx, spec, policy, f"{spec.name}:{policy.digest()}")
            items.append(
                (
                    spec,
                    policy,
                    entry,
                    with_gaps(common, entry.config.site, [entry.config.fleet], cap),
                )
            )
    quiet = ctx.seeds.quiet["validation"][: sizes.val_quiet]
    per_seed = sum(
        tactic_cost_s(e.config.site, t, e.config.params) for _, _, e, ts in items for t in ts
    )
    fixed = sum(quiet_cost_s(e.config.fleet, e.config.params) for _, _, e, _ in items) * len(quiet)
    n = _fit(ctx.budget_sim_s("validation"), per_seed, fixed, sizes.val_seeds)
    plan = ctx.checkpoint.setdefault("plans", {}).setdefault("validation", {"n": n})
    ctx.save_checkpoint()
    seeds = ctx.seeds.intrusion["validation"][: int(plan["n"])]
    ctx.say(f"validation: {len(items)} candidates on {len(seeds)} seeds, {len(quiet)} quiet nights")
    done = ctx.run(
        [Request(e.config, tuple(ts), seeds, quiet) for _, _, e, ts in items],
        "validation",
        ctx.stage_stop("validation"),
    )
    seeds = seeds[:done]
    if not seeds:
        return {"complete": False, "reason": "cut before any seed finished"}
    # the baseline and the fix are never candidates, but their worst tactic must also be
    # named on validation seeds before it is scored on final ones
    refs = [baseline_entry(ctx), fix_entry(ctx)]
    ref_tactics = {e.label: with_gaps(common, e.config.site, [e.config.fleet], cap) for e in refs}
    ctx.run(
        [Request(e.config, tuple(ref_tactics[e.label]), seeds, quiet) for e in refs], "validation"
    )
    reference = {
        e.label: score(ctx, e, ref_tactics[e.label], seeds, quiet, "validation", "stand-in").summary
        for e in refs
    }
    rows: dict[str, list[dict[str, Any]]] = {s.name: [] for s in specs}
    for spec, policy, entry, tactics in items:
        s = score(ctx, entry, tactics, seeds, quiet, "validation", "stand-in").summary
        s["digest"] = policy.digest()
        s["policy"] = policy.describe()
        rows[spec.name].append(s)
    chosen = {name: _rank(candidates)[0] for name, candidates in rows.items() if candidates}
    best = max(c["objective"] for c in chosen.values())
    near_best = [c for c in chosen.values() if c["objective"] >= best - KNEE]
    recommended = min(near_best, key=lambda c: (c["cost_per_hour"], c["hardware"]))
    both = [c for c in chosen.values() if c["pd"] >= TARGET and c["worst_naive"]["pd"] >= TARGET]
    if both:
        recommended = min(both, key=lambda c: (c["cost_per_hour"], c["hardware"]))
    picks = [recommended["hardware"]]
    frontier = []
    top = -1.0
    for c in sorted(chosen.values(), key=lambda c: (c["cost_per_hour"], -c["objective"])):
        if c["objective"] > top:
            top = c["objective"]
            frontier.append(c)
    for c in sorted(frontier, key=lambda c: -c["objective"]):
        if c["hardware"] not in picks and len(picks) < sizes.max_picks:
            picks.append(c["hardware"])
    # every configuration on a cost frontier of any of the three numbers; the assumption stage
    # runs these only, because a dominated configuration cannot be the cheapest to a target
    on_frontier = {recommended["hardware"]}
    metrics: dict[str, dict[str, float]] = {"objective": {}, "pd": {}, "worst": {}}
    for c in chosen.values():
        metrics["objective"][c["hardware"]] = float(c["objective"])
        metrics["pd"][c["hardware"]] = float(c["pd"])
        metrics["worst"][c["hardware"]] = float(c["worst_naive"]["pd"])
    for values in metrics.values():
        top = -1.0
        for c in sorted(chosen.values(), key=lambda c: (c["cost_per_hour"], c["hardware"])):
            if values[c["hardware"]] > top:
                top = values[c["hardware"]]
                on_frontier.add(c["hardware"])
    frontier_labels = sorted(
        BEST_FREE_LABEL if name == "d2_std_nocams" else f"best:{name}" for name in on_frontier
    )
    return {
        "complete": True,
        "n_seeds": len(seeds),
        "candidates": rows,
        "chosen": chosen,
        "reference": reference,
        "recommended": recommended["hardware"],
        "recommended_rule": (
            f"the cheapest configuration reaching {TARGET} on both metrics on validation seeds; "
            f"failing that, the cheapest within {KNEE} of the best selection objective"
        ),
        "picks": picks,
        "frontier_labels": frontier_labels,
    }


def chosen_entries(ctx: Ctx, validation: dict[str, Any]) -> list[Entry]:
    """The baseline, the fix, and the chosen policy of every hardware configuration."""
    out = [baseline_entry(ctx), fix_entry(ctx)]
    for spec in specs_for(ctx):
        row = validation["chosen"].get(spec.name)
        if row is not None:
            label = BEST_FREE_LABEL if spec.name == "d2_std_nocams" else f"best:{spec.name}"
            out.append(policy_entry(ctx, spec, _policy_from(row["policy"]), label))
    return out


INGREDIENTS = {
    "weight": ("asset_gain", "entry_gain", "band_gain", "weight_scale_m"),
    "controller": ("d0_m", "retarget_period_s", "top_fraction"),
    "schedule": ("offsets",),
    "docks": ("docks",),
}
INGREDIENT_PREFIX = "ingredient:"


def ingredient_entries(ctx: Ctx, validation: dict[str, Any]) -> list[Entry]:
    """The best free policy taken apart: the baseline's policy with one group of the best
    policy's fields at a time. Same agents, same cost, same hardware as the baseline."""
    row = validation["chosen"].get("d2_std_nocams")
    if row is None:
        return []
    best = _policy_from(row["policy"])
    plain = named_policy(ctx.baseline, "asset", 0.0, False)
    spec = _spec_by_name("d2_std_nocams")
    out = []
    for name, fields in INGREDIENTS.items():
        part = dataclasses.replace(plain, **{f: getattr(best, f) for f in fields}, origin=name)
        if part.digest() in (plain.digest(), best.digest()):
            continue  # this group does not differ from the baseline, or is the whole policy
        out.append(policy_entry(ctx, spec, part, f"{INGREDIENT_PREFIX}{name}"))
    return out


def final_entries(ctx: Ctx, validation: dict[str, Any]) -> list[Entry]:
    return [*chosen_entries(ctx, validation), *ingredient_entries(ctx, validation)]


def _final_set(ctx: Ctx, entries: Sequence[Entry]) -> list[Tactic]:
    return with_gaps(
        standin_common(ctx.site, ctx.limits, ctx.sizes.standin_phases),
        ctx.site,
        [e.config.fleet for e in entries],
        ctx.limits.speed_cap_mps,
    )


def _validation_worst(
    validation: dict[str, Any], entry: Entry, tactics: Sequence[Tactic]
) -> Tactic | None:
    if entry.label.startswith(INGREDIENT_PREFIX):
        return None  # an ingredient was never on validation seeds, so it has no such tactic
    if entry.label in (BASELINE_LABEL, FIX_LABEL):
        row = validation.get("reference", {}).get(entry.label)
    else:
        row = validation["chosen"].get(entry.hardware)
    if row is None:
        return None
    by_id = {t.id: t for t in tactics}
    return by_id.get(row["worst_naive"]["tactic"])


def stage_final_standin(
    ctx: Ctx, validation: dict[str, Any], stage: str = "final_standin", task_time_s: float = 0.0
) -> dict[str, Any]:
    """Every chosen configuration on final seeds against the stand-in set: the frontier. With
    task_time_s > 0 the same thing as an assumption run, never a headline."""
    sizes = ctx.sizes
    entries = final_entries(ctx, validation)
    tactics = _final_set(ctx, entries)
    if task_time_s > 0:
        keep = {BASELINE_LABEL, FIX_LABEL, BEST_FREE_LABEL, *validation["frontier_labels"]}
        entries = [e for e in entries if e.label in keep]
    quiet = ctx.seeds.quiet["final"][: sizes.final_quiet]

    def simulated(entry: Entry) -> Config:
        if task_time_s <= 0:
            return entry.config
        params = dataclasses.replace(entry.config.params, task_time_s=task_time_s)
        return dataclasses.replace(entry.config, params=params)

    per_seed = sum(
        tactic_cost_s(e.config.site, t, simulated(e).params) for e in entries for t in tactics
    )
    fixed = sum(quiet_cost_s(e.config.fleet, e.config.params) for e in entries) * len(quiet)
    n = _fit(
        ctx.budget_sim_s(stage), per_seed, fixed if task_time_s <= 0 else 0.0, sizes.final_seeds
    )
    plan = ctx.checkpoint.setdefault("plans", {}).setdefault(stage, {"n": n})
    ctx.save_checkpoint()
    seeds = ctx.seeds.intrusion["final"][: int(plan["n"])]
    ctx.say(f"{stage}: {len(entries)} configurations x {len(tactics)} tactics x {len(seeds)} seeds")
    stop = ctx.stage_stop(stage)
    ctx.run([Request(e.config, (), (), quiet) for e in entries], "final")
    done = ctx.run([Request(simulated(e), tuple(tactics), seeds) for e in entries], "final", stop)
    seeds = seeds[:done]
    if not seeds:
        return {"complete": False, "reason": "cut before any seed finished"}
    assumption = {"task_time_s": task_time_s} if task_time_s > 0 else None
    rows, scored_by_label = [], {}
    for entry in entries:
        worst = _validation_worst(validation, entry, tactics)
        scored = score(
            ctx,
            entry,
            tactics,
            seeds,
            quiet,
            "final",
            "stand-in",
            config=simulated(entry),
            assumption=assumption,
            heldout_worst=(worst, seeds) if worst is not None else None,
        )
        rows.append(scored.summary)
        scored_by_label[entry.label] = scored
    deltas = [
        paired(scored_by_label[e.label], scored_by_label[BASELINE_LABEL])
        for e in entries
        if e.label != BASELINE_LABEL
    ]
    upgrades = []
    by_hw = {e.hardware: e.label for e in entries if e.label not in (BASELINE_LABEL, FIX_LABEL)}
    for spec in specs_for(ctx):
        for bigger in _upgrades(spec):
            if spec.name in by_hw and bigger.name in by_hw:
                delta = paired(
                    scored_by_label[by_hw[bigger.name]], scored_by_label[by_hw[spec.name]]
                )
                delta["from"], delta["to"] = spec.name, bigger.name
                upgrades.append(delta)
    return {
        "complete": True,
        "entries": [record_of(e) for e in entries],
        "rows": rows,
        "paired_vs_baseline": deltas,
        "upgrades": upgrades,
        "task_time_s": task_time_s,
    }


def _upgrades(spec: hardware.HardwareSpec) -> list[hardware.HardwareSpec]:
    """The configurations one purchase above this one."""
    out = []
    if spec.n_drones + 1 in hardware.HARDWARE_DRONES:
        out.append(dataclasses.replace(spec, n_drones=spec.n_drones + 1))
    if not spec.swap_docks:
        out.append(dataclasses.replace(spec, swap_docks=True))
    if not spec.entry_cameras:
        out.append(dataclasses.replace(spec, entry_cameras=True))
    return out


def finalists(ctx: Ctx, validation: dict[str, Any]) -> list[Entry]:
    wanted = {BASELINE_LABEL, FIX_LABEL, BEST_FREE_LABEL}
    wanted |= {f"best:{name}" for name in validation["picks"]}
    if "d2_std_nocams" in validation["picks"]:
        wanted.add(BEST_FREE_LABEL)
    return [e for e in chosen_entries(ctx, validation) if e.label in wanted]


def stage_final_strong(ctx: Ctx, validation: dict[str, Any]) -> dict[str, Any]:
    return strong_evaluate(ctx, finalists(ctx, validation), "final_strong")


def recommended_without_cameras(validation: dict[str, Any]) -> str | None:
    """The recommended-configuration rule applied to the configurations that buy no entry
    camera, on validation numbers only. Camera configurations see every tactic at range 0 at
    the moment of entry, so the pitch also needs the best answer that does not lean on that."""
    chosen = [
        c for c in validation["chosen"].values() if not _spec_by_name(c["hardware"]).entry_cameras
    ]
    if not chosen:
        return None
    both = [c for c in chosen if c["pd"] >= TARGET and c["worst_naive"]["pd"] >= TARGET]
    if not both:
        best = max(c["objective"] for c in chosen)
        both = [c for c in chosen if c["objective"] >= best - KNEE]
    return str(min(both, key=lambda c: (c["cost_per_hour"], c["hardware"]))["hardware"])


def stage_final_strong_nocams(ctx: Ctx, validation: dict[str, Any]) -> dict[str, Any]:
    """The strong adversary against the recommended configuration WITHOUT entry cameras,
    paired with the baseline."""
    name = recommended_without_cameras(validation)
    if name is None:
        return {"complete": False, "reason": "no configuration without entry cameras"}
    label = BEST_FREE_LABEL if name == "d2_std_nocams" else f"best:{name}"
    wanted = {BASELINE_LABEL, label}
    entries = [e for e in chosen_entries(ctx, validation) if e.label in wanted]
    data = strong_evaluate(ctx, entries, "final_strong_nocams")
    data["recommended_without_cameras"] = name
    data["recommended_without_cameras_label"] = label
    data["rule"] = (
        "the recommended-configuration rule restricted to hardware without entry cameras, "
        "applied to validation numbers"
    )
    return data


def stage_sensitivity(ctx: Ctx, validation: dict[str, Any]) -> dict[str, Any]:
    """Task time and response time what-ifs for the baseline, the best free policy and the
    recommended configuration. Assumption rows only; nothing here is a headline."""
    sizes = ctx.sizes
    recommended = validation["recommended"]
    rec_label = BEST_FREE_LABEL if recommended == "d2_std_nocams" else f"best:{recommended}"
    all_entries = final_entries(ctx, validation)
    entries = [e for e in all_entries if e.label in (BASELINE_LABEL, BEST_FREE_LABEL, rec_label)]
    tactics = _final_set(ctx, all_entries)
    quiet = ctx.seeds.quiet["final"][: sizes.final_quiet]
    # two axes through the strict cell, not the full grid: the budget buys more seeds this way
    strict_resp = float(ctx.site.response_time_s)
    cells = [(task, strict_resp) for task in sizes.sens_task_times_s]
    cells += [(0.0, resp) for resp in sizes.sens_response_times_s if resp != strict_resp]

    def variant(entry: Entry, task: float, resp: float) -> Config:
        site = entry.config.site.model_copy(update={"response_time_s": resp})
        params = dataclasses.replace(entry.config.params, task_time_s=task)
        return Config(entry.config.name, site, entry.config.fleet, params)

    per_seed = sum(
        tactic_cost_s(e.config.site, t, variant(e, task, resp).params)
        for e in entries
        for task, resp in cells
        for t in tactics
    )
    n = _fit(ctx.budget_sim_s("sensitivity"), per_seed, 0.0, sizes.sens_seeds)
    plan = ctx.checkpoint.setdefault("plans", {}).setdefault("sensitivity", {"n": n})
    ctx.save_checkpoint()
    seeds = ctx.seeds.intrusion["final"][: int(plan["n"])]
    ctx.say(f"sensitivity: {len(entries)} configurations x {len(cells)} cells x {len(seeds)} seeds")
    ctx.run([Request(e.config, (), (), quiet) for e in entries], "final")

    def quiet_variant(entry: Entry, resp: float) -> Config:
        """Whose quiet nights a cell is scored with: its own when the response time differs."""
        return entry.config if resp == strict_resp else variant(entry, 0.0, resp)

    ctx.run(
        [Request(quiet_variant(e, resp), (), (), quiet) for e in entries for _, resp in cells],
        "final",
    )
    requests = [
        Request(variant(e, task, resp), tuple(tactics), seeds)
        for e in entries
        for task, resp in cells
    ]
    done = ctx.run(requests, "final", ctx.stage_stop("sensitivity"))
    seeds = seeds[:done]
    if not seeds:
        return {"complete": False, "reason": "cut before any seed finished"}
    rows = []
    for entry in entries:
        for task, resp in cells:
            strict = task == 0.0 and resp == float(ctx.site.response_time_s)
            s = score(
                ctx,
                entry,
                tactics,
                seeds,
                quiet,
                "final",
                "stand-in",
                config=variant(entry, task, resp),
                assumption=None if strict else {"task_time_s": task, "response_time_s": resp},
                quiet_config=quiet_variant(entry, resp),
            ).summary
            s["task_time_s"], s["response_time_s"] = task, resp
            rows.append(s)
    return {"complete": True, "rows": rows, "recommended_label": rec_label}


def stage_audit(ctx: Ctx, results: dict[str, Any]) -> dict[str, Any]:
    sizes = ctx.sizes
    out: dict[str, Any] = {}

    # 1. re-run random cached rows and require bit-identical results
    rng = np.random.default_rng(list(AUDIT_KEY))
    known = sorted(ctx.configs_seen)
    checked, mismatches = 0, []
    while checked < sizes.n_audit_rows and known:
        config = ctx.configs_seen[known[int(rng.integers(len(known)))]]
        keys = [
            k for k in ctx.ev.cached_keys(config) if k[0] == QUIET_KEY or k[0] in ctx.tactics_seen
        ]
        if not keys:
            known.remove(config.key(ctx.curves))
            continue
        kind, seed = keys[int(rng.integers(len(keys)))]
        if kind == QUIET_KEY and checked % 20 != 0:
            continue  # a quiet night costs thirty episodes, so only one row in twenty is one
        cached = ctx.ev.row(config, kind, seed)
        if kind == QUIET_KEY:
            fresh = quiet_task((config.site, config.fleet, ctx.curves, seed, config.params))[0]
        else:
            task = (
                config.site,
                config.fleet,
                ctx.tactics_seen[kind],
                ctx.curves,
                (seed,),
                config.params,
            )
            fresh = episode_task(task)[0]
        if json.loads(json.dumps(fresh)) != cached:
            mismatches.append({"config": config.name, "kind": kind, "seed": seed})
        ctx.ev.forget(config)
        checked += 1
    out["rerun"] = {"checked": checked, "mismatches": mismatches}
    ctx.say(f"audit: {checked} cached rows re-run, {len(mismatches)} mismatches")

    # 2. thresholds on the floor, 3. detection above 0.99
    final_rows = [
        {**row, "stage": stage}
        for stage, key in (
            ("final_standin", "rows"),
            ("final_strong", "final"),
            ("final_strong_nocams", "final"),
            ("A", "final"),
        )
        for row in results.get(stage, {}).get(key, [])
    ]
    out["threshold_flags"] = [
        {k: r[k] for k in ("stage", "label", "adversary", "tau", "flag", "tau_on_floor")}
        for r in final_rows
        if r["flag"] != roc.FLAG_OK or r["tau_on_floor"]
    ]
    suspects = sorted(
        {r["label"] for r in final_rows if r["pd"] > 0.99 or r["worst_naive"]["pd"] > 0.99}
    )
    out["too_good"] = []
    validation = results.get("validation", {})
    if suspects and validation.get("complete"):
        entries = {e.label: e for e in chosen_entries(ctx, validation)}
        fresh_tactics = adversary.random_tactics(
            ctx.site, ctx.limits, sizes.n_fresh, FRESH_TACTIC_KEY
        )
        seeds = ctx.seeds.intrusion["final"][: sizes.fresh_seeds]
        quiet = ctx.seeds.quiet["final"][: sizes.final_quiet]
        todo = [entries[label] for label in suspects if label in entries]
        ctx.run(
            [Request(e.config, tuple(fresh_tactics), seeds, quiet) for e in todo],
            "final",
            ctx.stage_stop("audit"),
        )
        for entry in todo:
            try:
                s = score(ctx, entry, fresh_tactics, seeds, quiet, "final", "fresh random").summary
            except KeyError:
                continue
            out["too_good"].append(
                {
                    "label": entry.label,
                    "n_tactics": len(fresh_tactics),
                    "n_seeds": len(seeds),
                    "pd": s["pd"],
                    "pd_ci": s["pd_ci"],
                    "worst": s["worst_naive"],
                    "n_tactics_below_target": sum(v < TARGET for v in s["pd_by_tactic"].values()),
                }
            )

    # 4. upgrades that lower detection
    upgrades = results.get("final_standin", {}).get("upgrades", [])
    out["upgrades_that_lower_detection"] = [
        u for u in upgrades if u["pd_delta"] < 0 or u["worst_delta"] < 0
    ]

    # 5. selection against report: validation and final numbers further apart than their intervals allow
    out["in_sample_vs_held_out"] = []
    chosen = validation.get("chosen", {})
    for row in results.get("final_standin", {}).get("rows", []):
        val = chosen.get(row["hardware"])
        if val is None or row["label"] in (BASELINE_LABEL, FIX_LABEL):
            continue
        for metric, a, b in (
            ("pd", (val["pd"], val["pd_ci"]), (row["pd"], row["pd_ci"])),
            (
                "worst",
                (val["worst_naive"]["pd"], val["worst_naive"]["ci"]),
                (row["worst_naive"]["pd"], row["worst_naive"]["ci"]),
            ),
        ):
            half = (a[1][1] - a[1][0]) / 2 + (b[1][1] - b[1][0]) / 2
            if abs(a[0] - b[0]) > half:
                out["in_sample_vs_held_out"].append(
                    {
                        "label": row["label"],
                        "metric": metric,
                        "validation": a[0],
                        "final": b[0],
                        "allowed": half,
                    }
                )
    return out


# ---- driver ----------------------------------------------------------------------------------


def cheapest_to_target(
    rows: Sequence[dict[str, Any]], metric: str, target: float = TARGET
) -> dict[str, Any]:
    """The cheapest row reaching the target on a metric ("pd", "worst_naive" or
    "worst_heldout"), else the ceiling: the best row and what it reached."""

    def value(row: dict[str, Any]) -> float | None:
        if metric == "pd":
            return float(row["pd"])
        return float(row[metric]["pd"]) if metric in row else None

    def interval(row: dict[str, Any]) -> list[float]:
        return list(row["pd_ci"] if metric == "pd" else row[metric]["ci"])

    known = [(r, v) for r, v in ((r, value(r)) for r in rows) if v is not None]
    reached = [(r, v) for r, v in known if v >= target]
    note = (
        "chosen and reported on the same final seeds, so the value flatters the choice; "
        "'confirmed' asks the lower end of the interval to reach the target as well"
    )
    if reached:
        row, v = min(reached, key=lambda rv: (rv[0]["cost_per_hour"], rv[0]["label"]))
        sure = [(r, x) for r, x in reached if interval(r)[0] >= target]
        confirmed = (
            min(sure, key=lambda rv: (rv[0]["cost_per_hour"], rv[0]["label"]))[0] if sure else None
        )
        return {
            "reached": True,
            "label": row["label"],
            "cost_per_hour": row["cost_per_hour"],
            "value": v,
            "ci": interval(row),
            "n_seeds": row["n_seeds"],
            "confirmed_label": confirmed["label"] if confirmed else None,
            "confirmed_cost_per_hour": confirmed["cost_per_hour"] if confirmed else None,
            "note": note,
        }
    if not known:
        return {"reached": False, "ceiling": None}
    row, v = max(known, key=lambda rv: (rv[1], -rv[0]["cost_per_hour"]))
    return {
        "reached": False,
        "ceiling": v,
        "ci": interval(row),
        "n_seeds": row["n_seeds"],
        "label": row["label"],
        "cost_per_hour": row["cost_per_hour"],
        "worst_tactic_of_ceiling": row["worst_naive"]["tactic"],
        "note": note,
    }


def git_commit(root: Path) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="absolute path to data/campaign")
    parser.add_argument("--deadline-hours", type=float, default=6.0)
    parser.add_argument(
        "--smoke", action="store_true", help="every stage at toy size, in <out>/smoke"
    )
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument("--tactics-dir", type=Path, action="append", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--frozen-root", type=Path, default=None, help="the worktree the code must run from"
    )
    parser.add_argument("--allow-new-code", action="store_true")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="run every stage again over the cache with the plans already made: scoring and "
        "selection are recomputed, and only what the cache lacks is simulated",
    )
    parser.add_argument("--stages", default=",".join(STAGES), help="comma separated; for debugging")
    args = parser.parse_args(argv)

    if not args.out.is_absolute():
        parser.error("--out must be an absolute path")
    data_dir = args.out.parent
    out = args.out / "smoke" if args.smoke else args.out
    code_root = Path(airtight.__file__).resolve()
    if args.frozen_root is not None:
        frozen = args.frozen_root.resolve()
        if frozen not in code_root.parents:
            raise SystemExit(
                f"airtight is imported from {code_root}, outside the frozen copy {frozen}"
            )
        commit = git_commit(frozen)
    else:
        commit = "working-tree:" + git_commit(code_root.parents[2])

    checkpoint_path = out / "checkpoint.json"
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("launch_commit") != commit and not args.allow_new_code:
            raise SystemExit(
                f"the checkpoint was written by {checkpoint.get('launch_commit')}, this is {commit}; "
                "pass --allow-new-code only if the new code cannot change a simulation result"
            )
        if checkpoint.get("engine_version") != ENGINE_VERSION:
            raise SystemExit(
                "the checkpoint was written by a different ENGINE_VERSION; start clean"
            )
    now = time.time()
    checkpoint.setdefault("launch_commit", commit)
    checkpoint.setdefault("commits", [])
    if commit not in checkpoint["commits"]:
        checkpoint["commits"].append(commit)
    checkpoint.setdefault("engine_version", ENGINE_VERSION)
    checkpoint.setdefault("started_at", now)
    checkpoint.setdefault("deadline_at", now + args.deadline_hours * 3600.0)
    checkpoint.setdefault("stages_done", [])
    if args.rescore:
        checkpoint["rescored_at"] = now
        checkpoint["stages_done"] = []
        checkpoint["started_at_rescore"] = now
        checkpoint["deadline_at_rescore"] = now + args.deadline_hours * 3600.0
    checkpoint.setdefault("wall_s", 0.0)
    checkpoint.setdefault("episodes", 0)
    checkpoint.setdefault("quiet_nights", 0)

    scenario_dir = args.scenario_dir or data_dir / "lane_c_export" / "scenarios" / "logistics_yard"
    tactic_dirs = args.tactics_dir or [data_dir / "lane_c_export" / "results"]
    site = Site.model_validate_json((scenario_dir / "site.json").read_text())
    curves = SensorCurves.model_validate_json((scenario_dir / "sensor_curve.json").read_text())
    spec = json.loads((scenario_dir / "fleets" / "sweep.json").read_text())
    baseline = FleetConfig.model_validate_json(
        (scenario_dir / "fleets" / f"{spec['baseline']}.json").read_text()
    )
    limits = adversary.load_limits(scenario_dir)
    lane_c, notes = adversary.load_tactic_files(
        [p for p in tactic_dirs if p.exists()], site, limits
    )
    source = data_dir / "lane_c_export" / "SOURCE_COMMIT"
    checkpoint["scenario_source_commit"] = (
        source.read_text().strip() if source.is_file() else "unknown"
    )
    checkpoint["lane_c_tactic_notes"] = notes

    sizes = SMOKE if args.smoke else FULL
    lane_c = lane_c[: sizes.max_lane_c] if sizes.max_lane_c is not None else lane_c
    for name in (PARAMS_JSON_ENV, TASK_TIME_ENV, "AIRTIGHT_WEIGHT_MODE"):
        if os.environ.get(name):
            raise SystemExit(f"{name} is set; the campaign builds every policy itself, unset it")
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    hardware.write_variants(
        args.out / "variants",
        baseline,
        site,
        curves,
        [s for s in hardware.all_specs() if sizes.specs is None or s.name in sizes.specs],
    )
    split = seedsplit.load(args.out / "seeds.json", create=True)
    with Evaluator(out / "cache", curves, split, workers) as ev:
        ctx = Ctx(
            out=out,
            sizes=sizes,
            site=site,
            curves=curves,
            baseline=baseline,
            limits=limits,
            costs=hardware.load_costs(args.out / "costs.json"),
            seeds=split,
            ev=ev,
            lane_c_tactics=lane_c,
            started=float(checkpoint["started_at_rescore" if args.rescore else "started_at"]),
            deadline=float(checkpoint["deadline_at_rescore" if args.rescore else "deadline_at"]),
            checkpoint=checkpoint,
        )
        ctx.say(
            f"campaign at {commit}; out {out}; {workers} workers; {len(lane_c)} lane C tactics; "
            f"deadline {time.strftime('%Y-%m-%d %H:%M', time.localtime(ctx.deadline))}"
        )
        ctx.save_checkpoint()
        results: dict[str, Any] = {}
        if args.rescore and (out / "stages").is_dir():
            keep_dir = out / f"stages_before_rescore_{int(now)}"
            shutil.copytree(out / "stages", keep_dir)
            ctx.say(f"rescore: the stage files as they were are kept in {keep_dir}")
        for stage in STAGES:
            path = out / "stages" / f"{stage}.json"
            if path.is_file():
                results[stage] = json.loads(path.read_text())
        wanted = [s for s in args.stages.split(",") if s]
        session_start = time.time()
        try:
            for stage in STAGES:
                if stage not in wanted:
                    continue
                if stage in checkpoint["stages_done"] and stage in results and stage != "probe":
                    ctx.say(f"{stage}: already done, skipped")
                    continue
                if stage != "probe" and time.time() > ctx.deadline:
                    ctx.say(f"{stage}: not started, the deadline has passed")
                    continue
                validation = results.get("validation", {})
                if stage in (
                    "final_standin",
                    "assumption60",
                    "final_strong",
                    "sensitivity",
                    "final_strong_nocams",
                ) and not validation.get("complete"):
                    ctx.say(f"{stage}: skipped, validation is not complete")
                    continue
                if stage == "validation" and "search" not in results:
                    ctx.say("validation: skipped, the search has no result")
                    continue
                began = time.time()
                try:
                    if stage == "probe":
                        data = stage_probe(ctx)
                    elif stage == "A":
                        data = stage_a(ctx)
                    elif stage == "search":
                        data = stage_search(ctx)
                    elif stage == "validation":
                        data = stage_validation(ctx, results["search"])
                    elif stage == "final_standin":
                        data = stage_final_standin(ctx, validation)
                    elif stage == "assumption60":
                        data = stage_final_standin(
                            ctx, validation, "assumption60", ASSUMED_TASK_TIME_S
                        )
                    elif stage == "final_strong":
                        data = stage_final_strong(ctx, validation)
                    elif stage == "sensitivity":
                        data = stage_sensitivity(ctx, validation)
                    elif stage == "final_strong_nocams":
                        data = stage_final_strong_nocams(ctx, validation)
                    else:
                        data = stage_audit(ctx, results)
                except CacheFull:
                    raise
                except Exception:  # noqa: BLE001  one broken stage must not cost the night
                    trace = traceback.format_exc()
                    ctx.say(f"{stage}: FAILED and skipped\n{trace}")
                    checkpoint.setdefault("failed_stages", {})[stage] = trace
                    ctx.save_checkpoint()
                    if stage == "probe":
                        raise
                    continue
                data["wall_s"] = time.time() - began
                results[stage] = data
                _write_json(out / "stages" / f"{stage}.json", data)
                if data.get("complete", True) and stage not in checkpoint["stages_done"]:
                    checkpoint["stages_done"].append(stage)
                ctx.say(
                    f"{stage}: finished in {data['wall_s'] / 60:.1f} min; cache "
                    f"{ev.cache_bytes / 1e6:.0f} MB; {ev.n_episodes} episodes so far this session"
                )
                ctx.save_checkpoint()
        except CacheFull as err:
            ctx.say(f"stopped: {err}")
            checkpoint["cache_full"] = True
        checkpoint["wall_s"] += time.time() - session_start
        checkpoint["episodes"] += ev.n_episodes
        checkpoint["quiet_nights"] += ev.n_quiet
        ctx.save_checkpoint()
        summary = {
            "checkpoint": checkpoint,
            "stages": results,
            "targets": _targets(results),
            "seed_split": split.describe(),
            "costs": ctx.costs,
            "engine_ignores": list(ENGINE_IGNORES),
            "constants": {
                "target": TARGET,
                "knee": KNEE,
                "far_target_per_hour": FAR_TARGET,
                "assumed_task_time_s": ASSUMED_TASK_TIME_S,
                "strict_response_time_s": float(site.response_time_s),
                "intruder_speed_min_mps": limits.speed_min_mps,
                "intruder_speed_cap_mps": limits.speed_cap_mps,
            },
            "finished_at": time.time(),
            "deadline_passed": time.time() > ctx.deadline,
            "shares": SHARES,
        }
        _write_json(out / "results.json", summary)
        ctx.say(f"results written to {out / 'results.json'}; CAMPAIGN FINISHED")
    return 0


def _targets(results: dict[str, Any]) -> dict[str, Any]:
    """cheapest_to_target(0.95) four ways, from final-seed rows only."""
    strict = results.get("final_standin", {}).get("rows", [])
    assumed = results.get("assumption60", {}).get("rows", [])
    strong = results.get("final_strong", {}).get("final", [])
    out: dict[str, Any] = {}
    for name, rows in (
        ("strict", strict),
        ("task_time_60s_assumption", assumed),
        ("strict_strong_finalists", strong),
        ("strict_strong_without_cameras", results.get("final_strong_nocams", {}).get("final", [])),
    ):
        rows = [r for r in rows if not str(r["label"]).startswith(INGREDIENT_PREFIX)]
        if rows:
            out[name] = {
                "overall": cheapest_to_target(rows, "pd"),
                "worst_tactic_naive": cheapest_to_target(rows, "worst_naive"),
                "worst_tactic_heldout": cheapest_to_target(rows, "worst_heldout"),
            }
    return out


if __name__ == "__main__":
    sys.exit(main())
