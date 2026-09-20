"""Hand the campaign's result to the other lanes: files, replays, a note, the contract report.

    python -m airtight.score.campaign_handoff --out /abs/path/data/campaign

Reads <out>/results.json and <out>/cache and never simulates a scored number. Four outputs,
under <out> or under --handoff-dir when that is given:

1. recommended/ and best_free_policy/: the fleet (charge offsets applied), the site variant
   when the hardware adds cameras, params.json for AIRTIGHT_PARAMS_JSON and env.json. The
   params file is read back through official_params() and must equal the policy's parameters.
2. replays/: full logs, baseline and recommended, of up to five final seeds where the baseline
   misses and the recommended configuration catches on the BASELINE's worst tactic, each at
   its own operating threshold, found in the cache. run_episode's verdict is read at TAU_REF,
   not at the operating threshold, so index.json holds both and says where they agree. Every
   log is parsed by lane A's plan_replay and the result is recorded.
3. NOTE_FOR_LANE_C.md in --note-dir, outside the repository: how to attack the recommended
   configuration again and where to put what is found.
4. contract_report/: data/report.json plus the recommended configuration, only when its paired
   held-out interval for worst-tactic detection against the baseline excludes zero. What the
   contract cannot say goes to the sidecar and a README, never into a bent field.

When the campaign's optional stage final_strong_nocams is complete, the camera-free
recommendation gets the same treatment: recommended_without_cameras/, its own pairs in
replays_without_cameras/, first place in the note (a camera on every entry sees every tactic
from range 0, so the camera result is an upper bound a tactic search cannot hurt), and its own
gate for the contract report. Without that stage everything else is unchanged.

The replays are selected ON the final seeds, so they are illustrations. They are not evidence
and nothing about them is reported as a confirmed number.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from airtight.contracts import ConfigResult, FleetConfig, Report, RocPoint, SensorCurves, Site
from airtight.score import adversary, campaign, roc, seedsplit
from airtight.score.cells import QUIET_KEY, Evaluator
from airtight.score.config_score import peaks_and_quiet, score_config
from airtight.score.report import FAR_TARGET, METRIC_WORST, ConfigDetail, paired_deltas
from airtight.score.sweep import fleet_gaps
from airtight.sim.constants import NEVER_SEEN, TAU_REF, TIME_EPS
from airtight.sim.coverage import coverage_profile, uncovered_s_per_hour
from airtight.sim.episode import (
    PARAMS_JSON_ENV,
    TASK_TIME_ENV,
    EpisodeParams,
    EpisodeScores,
    QuietScores,
    official_params,
    params_to_mapping,
)
from airtight.sim.runner import ENGINE_ENV, run_episode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from airtight.contracts import Tactic
    from airtight.score.campaign import Ctx, Entry
    from airtight.score.cells import Config

REPLAY_ENGINE = "v0"
WEIGHT_MODE_ENV = "AIRTIGHT_WEIGHT_MODE"
POLICY_ENVS = (ENGINE_ENV, PARAMS_JSON_ENV, WEIGHT_MODE_ENV, TASK_TIME_ENV)
MAX_PAIRS = 5
STRONG, STANDIN = "final_strong", "final_standin"
NOCAMS = "final_strong_nocams"
MAIN_STAGES = (STRONG, STANDIN)
REC_ROLE, NOCAMS_ROLE = "recommended", "recommended_without_cameras"
DEFAULT_NOTE_DIR = Path("/Users/rishabghosh/Projects/AirTight-lane-c-note")
NOTE_NAME = "NOTE_FOR_LANE_C.md"
ATTACKS_DIR = Path("lane_c_export") / "results" / "attacks_on_recommended"
LANE_A_REPLAY = "airtight.dimos_lane.replay"


class HandoffError(RuntimeError):
    """Something the handoff needs is missing or does not match. Reported, never fatal."""


class RowReader(Protocol):
    def row(self, config: Config, kind: str, seed: int) -> dict[str, Any] | None: ...


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@contextlib.contextmanager
def policy_env(engine: str | None, params_json: Path | None) -> Iterator[None]:
    """The engine and the policy file for the block, every other policy variable unset, and
    the environment exactly as it was afterwards."""
    saved = {name: os.environ.get(name) for name in POLICY_ENVS}
    try:
        for name in POLICY_ENVS:
            os.environ.pop(name, None)
        if engine is not None:
            os.environ[ENGINE_ENV] = engine
        if params_json is not None:
            os.environ[PARAMS_JSON_ENV] = str(params_json)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---- reading the campaign --------------------------------------------------------------------


def recommended_label(stages: Mapping[str, Any]) -> str | None:
    validation = stages.get("validation", {})
    if not validation.get("complete") or not validation.get("recommended"):
        return None
    name = str(validation["recommended"])
    return campaign.BEST_FREE_LABEL if name == "d2_std_nocams" else f"best:{name}"


def stage_rows(stage: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    rows = stage.get("rows" if name == STANDIN else "final", [])
    return list(rows) if stage.get("complete") else []


def nocams_label(stages: Mapping[str, Any]) -> str | None:
    """The camera-free recommendation's label, when its optional stage is complete."""
    stage = stages.get(NOCAMS, {})
    label = stage.get("recommended_without_cameras_label")
    return str(label) if stage.get("complete") and label else None


def pick_stage(
    stages: Mapping[str, Any], labels: Sequence[str], names: Sequence[str] = MAIN_STAGES
) -> tuple[str, dict[str, dict[str, Any]]] | None:
    """The first of the named stages (by default the strong one, then the stand-in one) that
    is complete and holds every label, else None. Returns its name and its rows by label."""
    for name in names:
        by_label = {str(r["label"]): r for r in stage_rows(stages.get(name, {}), name)}
        if all(label in by_label for label in labels):
            return name, by_label
    return None


def baseline_worst_tactic(row: Mapping[str, Any]) -> tuple[str, str]:
    """(tactic id, which estimate named it) for a final row."""
    held = row.get("worst_heldout")
    if held:
        return str(held["tactic"]), "worst_heldout"
    return str(row["worst_naive"]["tactic"]), "worst_naive"


def gate(
    stages: Mapping[str, Any], label: str, names: Sequence[str] = MAIN_STAGES
) -> dict[str, Any]:
    """The criterion of fix.verdict_for on the campaign's own paired rows: the recommended
    configuration enters the contract report only if the paired interval for worst-tactic
    detection against the baseline, on final seeds, lies above zero."""
    for name in names:
        stage = stages.get(name, {})
        if not stage.get("complete"):
            continue
        for delta in stage.get("paired_vs_baseline", []):
            if delta.get("a") == label and delta.get("b") == campaign.BASELINE_LABEL:
                lo, hi = (float(v) for v in delta["worst_delta_ci"])
                return {
                    "stage": name,
                    "passed": lo > 0,
                    "worst_delta": float(delta["worst_delta"]),
                    "worst_delta_ci": [lo, hi],
                    "n_seeds": delta.get("n_seeds"),
                    "reason": "" if lo > 0 else "the paired interval does not exclude zero",
                }
    return {
        "stage": None,
        "passed": False,
        "worst_delta": None,
        "worst_delta_ci": None,
        "n_seeds": None,
        "reason": f"no complete stage among {list(names)} holds a paired row for {label!r}",
    }


def entries_of(ctx: Ctx, stage: Mapping[str, Any]) -> dict[str, Entry]:
    """The stage's configurations, rebuilt from their records and checked against the hashes
    the campaign wrote, so a replay can never be of a different configuration."""
    out: dict[str, Entry] = {}
    for record in stage.get("entries", []):
        entry = campaign.entry_from_record(ctx, record)
        rebuilt = campaign.record_of(entry)
        for key in ("fleet_hash", "site_hash", "config"):
            if rebuilt[key] != record[key]:
                raise HandoffError(
                    f"{record['label']}: rebuilt {key} {rebuilt[key]!r} is not the campaign's "
                    f"{record[key]!r}; the scenario or the code has changed since the run"
                )
        out[entry.label] = entry
    return out


def stage_tactics(
    ctx: Ctx,
    name: str,
    entries: Sequence[Entry],
    wanted_ids: set[str],
    all_lane_c: Sequence[Tactic] | None = None,
) -> list[Tactic]:
    """The stage's tactic set in the campaign's own order, rebuilt without simulating. For
    the strong stage the kept random tactics are recognised by id among the keyed draw."""
    if name == STANDIN:
        candidates = [campaign._final_set(ctx, entries)]
    else:
        randoms = adversary.random_tactics(
            ctx.site, ctx.limits, ctx.sizes.n_random, campaign.RANDOM_TACTIC_KEY
        )
        kept = sorted((t for t in randoms if t.id in wanted_ids), key=lambda t: t.id)
        candidates = []
        for lane_c in (ctx.lane_c_tactics, all_lane_c or ()):
            base = [
                *adversary.strong_grid(ctx.site, ctx.limits, ctx.sizes.strong_phases),
                *(t for t in lane_c if t.id in wanted_ids),
            ]
            base = campaign.with_gaps(
                base, ctx.site, [e.config.fleet for e in entries], ctx.limits.speed_cap_mps
            )
            candidates.append([*base, *kept])
    for tactics in candidates:
        if {t.id for t in tactics} == wanted_ids and len(tactics) == len(wanted_ids):
            return tactics
    got = {t.id for t in candidates[0]}
    raise HandoffError(
        f"{name}: the rebuilt tactic set does not match the stage's: "
        f"{len(wanted_ids - got)} missing (first {sorted(wanted_ids - got)[:3]}), "
        f"{len(got - wanted_ids)} extra (first {sorted(got - wanted_ids)[:3]})"
    )


# ---- 1. configuration files ------------------------------------------------------------------


def verify_params_file(path: Path, expected: EpisodeParams) -> None:
    """official_params() with AIRTIGHT_PARAMS_JSON on this file must be the policy's own."""
    with policy_env(None, path):
        got = official_params()
    if got != expected:
        raise HandoffError(f"{path} does not read back as the policy's parameters")


def write_config_files(entry: Entry, base_site: Site, out_dir: Path) -> dict[str, Any]:
    """Fleet, site variant when it differs, params.json, env.json. Returns what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = entry.config
    fleet_path = out_dir / "fleet.json"
    fleet_path.write_text(config.fleet.model_dump_json(indent=2) + "\n")
    FleetConfig.model_validate_json(fleet_path.read_text())
    site_path: Path | None = None
    if config.site.content_hash() != base_site.content_hash():
        site_path = out_dir / "site.json"
        site_path.write_text(config.site.model_dump_json(indent=2) + "\n")
        Site.model_validate_json(site_path.read_text())
    params_path = out_dir / "params.json"
    if entry.policy is None:
        mapping = params_to_mapping(config.params)
    else:
        mapping = campaign._policy_from(dict(entry.policy)).params_json()
    _write_json(params_path, mapping)
    verify_params_file(params_path.resolve(), config.params)
    base_ids = {s.id for s in base_site.fixed_sensors}
    written: dict[str, Any] = {
        "label": entry.label,
        "config": config.name,
        "hardware": entry.hardware,
        "cost_per_hour": entry.cost_per_hour,
        "fleet": str(fleet_path.resolve()),
        "fleet_hash": config.fleet.content_hash(),
        "site_variant": None if site_path is None else str(site_path.resolve()),
        "site_hash": config.site.content_hash(),
        "added_fixed_sensors": [s.id for s in config.site.fixed_sensors if s.id not in base_ids],
        "params_json": str(params_path.resolve()),
        "params_verified_through_official_params": True,
        "env": {ENGINE_ENV: REPLAY_ENGINE, PARAMS_JSON_ENV: str(params_path.resolve())},
        "must_be_unset": [WEIGHT_MODE_ENV, TASK_TIME_ENV],
        "policy": entry.policy,
    }
    _write_json(out_dir / "env.json", written)
    return written


# ---- 2. replays ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    seed: int
    baseline_peak: float
    recommended_peak: float
    baseline_timely_at_ref: bool
    recommended_timely_at_ref: bool

    @property
    def also_in_log(self) -> bool:
        """Miss-then-catch at TAU_REF too, which is what the log's own verdict shows."""
        return self.recommended_timely_at_ref and not self.baseline_timely_at_ref


def _timely_at_ref(row: Mapping[str, Any]) -> bool:
    t_alarm = row.get("intruder_t_alarm_ref")
    return t_alarm is not None and float(t_alarm) <= float(row["t_cdp"]) + TIME_EPS


def find_pairs(
    reader: RowReader,
    baseline: Config,
    recommended: Config,
    tactic: Tactic,
    seeds: Sequence[int],
    tau_baseline: float,
    tau_recommended: float,
) -> tuple[list[Pair], int]:
    """Seeds, in list order, where the baseline misses and the recommended configuration
    catches, each at its own operating threshold, from cached rows only. Also returns how
    many seeds had both rows."""
    kind = tactic.content_hash()
    pairs, examined = [], 0
    for seed in seeds:
        b, r = reader.row(baseline, kind, seed), reader.row(recommended, kind, seed)
        if b is None or r is None:
            continue
        examined += 1
        b_peak, r_peak = float(b["intruder_peak"]), float(r["intruder_peak"])
        missed = b_peak < tau_baseline or b_peak <= NEVER_SEEN
        caught = r_peak >= tau_recommended and r_peak > NEVER_SEEN
        if missed and caught:
            pairs.append(Pair(seed, b_peak, r_peak, _timely_at_ref(b), _timely_at_ref(r)))
    return pairs, examined


def order_pairs(pairs: Sequence[Pair], n: int = MAX_PAIRS) -> list[Pair]:
    """Pairs that are also miss-then-catch in the log first, seed-list order within each."""
    return [*(p for p in pairs if p.also_in_log), *(p for p in pairs if not p.also_in_log)][:n]


def check_log(path: Path) -> dict[str, Any]:
    """Lane A's plan_replay on one log. dispatch=False: the parse is what is being checked,
    and the dispatch path would drive lane A's fleet backend."""
    try:
        plan_replay: Callable[..., Any] = importlib.import_module(LANE_A_REPLAY).plan_replay
    except Exception as err:  # noqa: BLE001  any import failure is a reason, never a crash
        return {"ok": None, "reason": f"plan_replay is not importable here: {err!r}"}
    try:
        plan = plan_replay(path, dispatch=False)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "reason": f"plan_replay raised {err!r}"}
    return {
        "ok": True,
        "dispatch": False,
        "title": plan.title,
        "seed": plan.seed,
        "tactic_id": plan.tactic_id,
        "timely_detected": plan.timely_detected,
        "n_intruder_poses": len(plan.intruder),
        "markers": sorted(plan.markers),
        "dispatch_at": plan.dispatch_at,
    }


def export_replays(
    ctx: Ctx,
    stages: Mapping[str, Any],
    label: str,
    rec_files: Mapping[str, Any],
    out_dir: Path,
    all_lane_c: Sequence[Tactic] | None = None,
    names: Sequence[str] = MAIN_STAGES,
) -> dict[str, Any]:
    picked = pick_stage(stages, [campaign.BASELINE_LABEL, label], names)
    if picked is None:
        raise HandoffError(
            f"no stage among {list(names)} is complete with both the baseline and {label!r}"
        )
    name, rows = picked
    base_row, rec_row = rows[campaign.BASELINE_LABEL], rows[label]
    entries = entries_of(ctx, stages[name])
    base, rec = entries[campaign.BASELINE_LABEL], entries[label]
    tactic_id, source = baseline_worst_tactic(base_row)
    tactics = stage_tactics(
        ctx, name, list(entries.values()), set(base_row["pd_by_tactic"]), all_lane_c
    )
    tactic = next(t for t in tactics if t.id == tactic_id)
    n = int(base_row["n_seeds"])
    if source == "worst_heldout":
        n = max(n, int(base_row["worst_heldout"]["n_seeds"]))
    seeds = ctx.seeds.intrusion["final"][:n]
    tau_b, tau_r = float(base_row["tau"]), float(rec_row["tau"])
    found, examined = find_pairs(ctx.ev, base.config, rec.config, tactic, seeds, tau_b, tau_r)
    chosen = order_pairs(found)

    kind = tactic.content_hash()
    sides = (
        ("baseline", base, tau_b, None),
        ("recommended", rec, tau_r, Path(str(rec_files["params_json"]))),
    )
    pair_rows = []
    for pair in chosen:
        row: dict[str, Any] = {
            "seed": pair.seed,
            "tactic_id": tactic.id,
            "miss_then_catch_at_operating_thresholds": True,
        }
        for side, entry, tau, params_json in sides:
            log_dir = out_dir / side
            log_dir.mkdir(parents=True, exist_ok=True)
            config = entry.config
            with policy_env(REPLAY_ENGINE, params_json):
                outcome = run_episode(
                    config.site, config.fleet, tactic, ctx.curves, pair.seed, log_dir, full_log=True
                )
            cached = ctx.ev.row(config, kind, pair.seed) or {}
            peak = float(cached["intruder_peak"])
            row[side] = {
                "label": entry.label,
                "configuration": config.name,
                "params_json": None if params_json is None else str(params_json),
                "operating_threshold": tau,
                "intruder_peak_before_cdp": None if peak <= NEVER_SEEN else peak,
                "detected_at_operating_threshold": bool(peak >= tau and peak > NEVER_SEEN),
                "timely_detected_in_log_at_tau_ref": outcome.timely_detected,
                "t_alarm": outcome.t_alarm,
                "t_cdp": outcome.t_cdp,
                "log_agrees_with_cache": bool(
                    outcome.t_cdp == cached.get("t_cdp")
                    and outcome.t_alarm == cached.get("intruder_t_alarm_ref")
                ),
                "log_path": str(outcome.log_path.resolve()),
                "plan_replay": check_log(outcome.log_path),
            }
        row["miss_then_catch_in_the_log_itself"] = bool(
            row["recommended"]["timely_detected_in_log_at_tau_ref"]
            and not row["baseline"]["timely_detected_in_log_at_tau_ref"]
        )
        pair_rows.append(row)
    return {
        "complete": True,
        "stage": name,
        "recommended_label": label,
        "tactic": {"id": tactic.id, "chosen_by": f"the baseline row's {source} in {name}"},
        "final_seeds_examined": examined,
        "pairs_found": len(found),
        "pairs_found_that_are_also_miss_then_catch_at_tau_ref": sum(p.also_in_log for p in found),
        "pairs_exported": len(pair_rows),
        "tau_ref": TAU_REF,
        "note": (
            "A pair is a final seed where the baseline's peak track score before t_cdp is below "
            "the baseline's operating threshold and the recommended configuration's is at or "
            "above its own. The log's timely_detected is run_episode's verdict at TAU_REF, a "
            "different threshold, so the two can disagree; both are given per side. Pairs were "
            "picked on the final seeds: they illustrate, they are not evidence."
        ),
        "pairs": pair_rows,
    }


# ---- 3. the note for lane C ------------------------------------------------------------------

UPPER_BOUND = (
    "Its result is an UPPER BOUND, not a measurement of what cameras are worth. Each purchased "
    "camera stands exactly on an entry point, and every tactic starts on an entry point, so "
    "every intruder is looked at from range 0 at t = 0 whatever route it takes afterwards. The "
    "engine has 2D footprints, no occlusion, truth association and no way to blind, spoof or "
    "walk round a camera, so no choice of entry, route, speed or phase can avoid that first "
    "look. A search over tactics therefore cannot hurt this configuration much, and that says "
    "more about the model than about the cameras."
)


def _note_section(
    role: str, rec: Mapping[str, Any], scenario_dir: Path, repo_root: Path, attacks: Path
) -> list[str]:
    site = rec["site_variant"] or str(scenario_dir / "site.json")
    cameras = rec["added_fixed_sensors"]
    found = attacks / role
    return [
        f"Label `{rec['label']}`, configuration `{rec['config']}`, hardware `{rec['hardware']}`, "
        f"{rec['cost_per_hour']:.2f} USD per hour.",
        "",
        f"- Fleet file: `{rec['fleet']}`",
        f"- Site file: `{site}`"
        + (
            f" (a site VARIANT: the scenario's site plus the cameras {', '.join(cameras)}; it "
            "MUST be passed as the site, with the scenario's own site.json the cameras are absent)"
            if rec["site_variant"]
            else " (the scenario's own site; this hardware adds no camera)"
        ),
        f"- Policy parameters, the value of `{PARAMS_JSON_ENV}`: `{rec['params_json']}`",
        f"- Environment: `{ENGINE_ENV}={REPLAY_ENGINE}`, `{PARAMS_JSON_ENV}` as above, "
        f"`{WEIGHT_MODE_ENV}` and `{TASK_TIME_ENV}` unset (also in `env.json` next to the fleet)",
        "",
        "```",
        f"cd {repo_root}",
        f"unset {WEIGHT_MODE_ENV}",
        f"unset {TASK_TIME_ENV}",
        f"export {ENGINE_ENV}={REPLAY_ENGINE}",
        f"export {PARAMS_JSON_ENV}={rec['params_json']}",
        f".venv/bin/airtight-redteam search --engine {REPLAY_ENGINE} --site {site} "
        f"--fleet {rec['fleet']} --curves {scenario_dir / 'sensor_curve.json'} "
        f"--config {scenario_dir / 'redteam_config.json'} --out {found} "
        f"--log-dir {found / 'search_logs'} --workers 2",
        "```",
        "",
    ]


def lane_c_note(
    configs: Mapping[str, Mapping[str, Any]],
    campaign_root: Path,
    scenario_dir: Path,
    repo_root: Path,
) -> str:
    """configs maps a role (REC_ROLE, NOCAMS_ROLE) to what write_config_files returned. The
    camera-free configuration comes first: it is the one a tactic search can hurt."""
    attacks = campaign_root.parent / ATTACKS_DIR
    rerun_out = campaign_root.with_name(f"{campaign_root.name}_with_attacks")
    rec, nocams = configs.get(REC_ROLE), configs.get(NOCAMS_ROLE)
    rec_has_cameras = rec is not None and bool(rec["added_fixed_sensors"])
    lines = [
        "# The campaign's recommended configurations, for lane C to attack",
        "",
        "## Two things that silently give the wrong answer",
        "",
        f"1. run_episode applies the patrol policy ONLY when `{PARAMS_JSON_ENV}` is set to the "
        "configuration's params.json. The policy is a set of engine parameters, not fleet "
        "fields. With the variable unset you attack the default patrol with this fleet's "
        "charge offsets, which is a different and weaker configuration. Each configuration "
        "below has its OWN params.json: never reuse one for the other.",
        "2. When the hardware includes entry cameras, they exist only in the site variant, so "
        "the site variant MUST be passed as the site. With the scenario's own site.json the "
        "cameras are absent.",
        "",
    ]
    if nocams is not None:
        lines += [
            "## 1. The recommended configuration WITHOUT cameras: attack this one first",
            "",
            "This is where a new tactic can change the answer."
            + (" See below for why the camera configuration is not." if rec_has_cameras else ""),
            "",
            *_note_section(NOCAMS_ROLE, nocams, scenario_dir, repo_root, attacks),
        ]
    if rec is not None:
        same = nocams is not None and nocams["label"] == rec["label"]
        lines += [
            f"## {'2' if nocams is not None else '1'}. The main recommendation",
            "",
            *(
                ["It is the same configuration as the one above.", ""]
                if same
                else [UPPER_BOUND, ""]
                if rec_has_cameras
                else ["This hardware adds no camera.", ""]
            ),
            *_note_section(REC_ROLE, rec, scenario_dir, repo_root, attacks),
        ]
    if nocams is None:
        lines += [
            "No camera-free recommendation is on record (the campaign's optional stage "
            f"`{NOCAMS}` is missing or incomplete), so only the main one is given.",
            "",
        ]
    lines += [
        "## Where to put what you find",
        "",
        f"`{attacks}/`, any JSON layout, any subfolder (the commands above use one per "
        "configuration). The campaign's loader walks the whole results directory, takes every "
        "object that has `entry_id` and `waypoints`, validates it against the Tactic contract "
        "and the limits in redteam_config.json, and notes every rejection with its reason. The "
        "campaign then scores every tactic itself, on its own seeds, against every finalist.",
        "",
        "The one command that reruns the campaign with them (a fresh output directory, so no "
        "old checkpoint is resumed; the campaign refuses to start with a policy variable set):",
        "",
        "```",
        f"env -u {PARAMS_JSON_ENV} -u {WEIGHT_MODE_ENV} -u {TASK_TIME_ENV} "
        f".venv/bin/python -m airtight.score.campaign --deadline-hours 6 --out {rerun_out}",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---- 4. the contract report ------------------------------------------------------------------


def _scores_from_cache(
    ctx: Ctx,
    entry: Entry,
    tactics: Sequence[Tactic],
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
) -> tuple[dict[str, list[EpisodeScores]], list[QuietScores]]:
    episodes: dict[str, list[EpisodeScores]] = {}
    for tactic in tactics:
        kind = tactic.content_hash()
        found = [ctx.ev.row(entry.config, kind, seed) for seed in seeds]
        if any(row is None for row in found):
            raise HandoffError(f"{entry.label}: tactic {tactic.id!r} is not cached on every seed")
        episodes[tactic.id] = [EpisodeScores(**row) for row in found if row is not None]
    nights = [ctx.ev.row(entry.config, QUIET_KEY, seed) for seed in quiet_seeds]
    if any(row is None for row in nights):
        raise HandoffError(f"{entry.label}: a quiet night is not cached")
    return episodes, [QuietScores(**row) for row in nights if row is not None]


def config_for_report(
    ctx: Ctx,
    stages: Mapping[str, Any],
    label: str,
    verdict: Mapping[str, Any],
    taken: Sequence[str],
    all_lane_c: Sequence[Tactic] | None = None,
) -> tuple[ConfigResult, ConfigDetail, dict[str, Any]]:
    """One configuration as a contract ConfigResult, scored from the cache on the stage its
    gate used, the way fix.add_fix_to_report adds a confirmed fix. Also its sidecar entry and
    what the contract cannot say about it. taken holds the names the report already uses."""
    name = str(verdict["stage"])
    rows = {str(r["label"]): r for r in stage_rows(stages[name], name)}
    entries = entries_of(ctx, stages[name])
    base, rec = entries[campaign.BASELINE_LABEL], entries[label]
    if rec.config.name in taken:
        raise HandoffError(f"the report already holds a configuration named {rec.config.name!r}")
    base_row, rec_row = rows[campaign.BASELINE_LABEL], rows[label]
    tactics = stage_tactics(
        ctx, name, list(entries.values()), set(rec_row["pd_by_tactic"]), all_lane_c
    )
    seeds = ctx.seeds.intrusion["final"][: int(rec_row["n_seeds"])]
    quiet_seeds = ctx.seeds.quiet["final"][: int(rec_row["n_quiet"])]
    n_boot = ctx.sizes.n_boot

    scored = {}
    for entry in (rec, base):
        episodes, nights = _scores_from_cache(ctx, entry, tactics, seeds, quiet_seeds)
        score = score_config(episodes, nights, FAR_TARGET, n_boot)
        _, peaks, quiet = peaks_and_quiet(episodes, nights)
        scored[entry.label] = (score, roc.replicates(peaks, quiet, n_boot, 0, FAR_TARGET))
    score, reps = scored[label]
    base_score, base_reps = scored[campaign.BASELINE_LABEL]
    deltas = paired_deltas(score, reps, base_score, base_reps)
    worst = next(d for d in deltas if d.metric == METRIC_WORST)
    if not worst.ci[0] > 0:
        raise HandoffError(
            f"recomputed from the cache the paired worst-tactic interval is {list(worst.ci)}, "
            "which does not exclude zero, although results.json says it does"
        )
    agrees = {
        "tau": score.tau == rec_row["tau"] and base_score.tau == base_row["tau"],
        "pd": score.pd == rec_row["pd"] and base_score.pd == base_row["pd"],
        "worst_delta_ci": list(worst.ci) == list(verdict["worst_delta_ci"]),
    }

    fleet, site = rec.config.fleet, rec.config.site
    profile = coverage_profile(site, fleet, ctx.curves, rec.config.params, seed=seeds[0])
    result = ConfigResult(
        config_name=rec.config.name,
        fleet_hash=fleet.content_hash(),
        n_episodes=len(seeds) * len(tactics),
        roc=[
            RocPoint(threshold=r.tau, pd=r.pd, pd_ci=r.pd_ci, far_per_hour=r.far) for r in score.roc
        ],
        pd_at_operating_point=score.pd,
        pd_at_operating_point_ci=score.pd_ci,
        worst_tactic_id=score.worst_tactic_id,
        worst_tactic_pd=score.worst_tactic_pd,
        cost_per_hour=rec.cost_per_hour,
        coverage_gap_s_per_hour=uncovered_s_per_hour(profile),
        human_decisions_per_hour=score.human_decisions_per_hour,
        paired_vs_baseline=deltas,
    )
    gaps = fleet_gaps(fleet)
    phases = sorted({t.phase for t in tactics})
    config_detail = ConfigDetail(
        operating_threshold=score.tau,
        flag=score.flag,
        pd_by_tactic=score.pd_by_tactic,
        worst_tactic_pd_ci=score.worst_tactic_pd_ci,
        raw_alerts_per_hour=score.raw_alerts_per_hour,
        quiet_hours=score.quiet_hours,
        uncovered_phase_ranges=gaps["uncovered"],
        drones_down_phase_ranges=gaps["drones_down"],
        drones_down_s_per_hour=sum(b - a for a, b in gaps["drones_down"]) * 3600.0,
        tactic_phases_in_uncovered=sum(
            any(a <= p < b for a, b in gaps["uncovered"]) for p in phases
        ),
        tactic_phases_in_drones_down=sum(
            any(a <= p < b for a, b in gaps["drones_down"]) for p in phases
        ),
    )
    base_ids = {s.id for s in ctx.site.fixed_sensors}
    cameras = [s.id for s in site.fixed_sensors if s.id not in base_ids]
    cannot_say = [
        "Conditions.n_seeds and Conditions.seed_list_hash describe the sweep's seeds. This "
        f"row was scored on {len(seeds)} final seeds of the campaign's own list, against the "
        f"campaign's {name} tactic set ({len(tactics)} tactics), not the sweep's.",
        "paired_vs_baseline is against the baseline scored on those same campaign seeds and "
        "tactics (kept in this sidecar), not against the baseline row of this report.",
        "The patrol policy lives in engine parameters (AIRTIGHT_PARAMS_JSON). fleet_hash "
        "covers the charge offsets only, so the fleet alone does not reproduce this row.",
    ]
    if site.content_hash() != ctx.site.content_hash():
        cannot_say.append(
            f"Report.site_hash is the scenario's site. This configuration runs on a site "
            f"variant, hash {site.content_hash()}, which adds the cameras {cameras}."
        )
    if cameras:
        cannot_say.append(
            "The contract has no field for 'upper bound'. Every added camera stands on an "
            "entry point and every tactic starts on one, so the intruder is always looked at "
            "from range 0 at t = 0, and the engine has no occlusion and no way round a camera: "
            "this row's detection is an upper bound on what the cameras give."
        )
    if rec.cost_per_hour != fleet.cost_per_hour():
        cannot_say.append(
            f"cost_per_hour {rec.cost_per_hour} includes purchased hardware that is not in the "
            f"fleet file, whose own cost is {fleet.cost_per_hour()}."
        )
    extra = {
        "label": label,
        "config_name": rec.config.name,
        "hardware": rec.hardware,
        "policy": rec.policy,
        "params_json": params_to_mapping(rec.config.params),
        "site_hash": site.content_hash(),
        "added_fixed_sensors": cameras,
        "stage": name,
        "n_seeds": len(seeds),
        "n_quiet_seeds": len(quiet_seeds),
        "n_tactics": len(tactics),
        "n_boot": n_boot,
        "gate": dict(verdict),
        "recomputed_from_cache_equals_results_json": agrees,
        "baseline_on_the_same_seeds_and_tactics": {
            "operating_threshold": base_score.tau,
            "pd": base_score.pd,
            "pd_ci": list(base_score.pd_ci),
            "worst_tactic_id": base_score.worst_tactic_id,
            "worst_tactic_pd": base_score.worst_tactic_pd,
            "worst_tactic_pd_ci": list(base_score.worst_tactic_pd_ci),
            "human_decisions_per_hour": base_score.human_decisions_per_hour,
        },
        "worst_tactic_held_out": rec_row.get("worst_heldout"),
        "what_the_contract_cannot_say": cannot_say,
    }
    return result, config_detail, extra


def write_contract_report(
    ctx: Ctx,
    stages: Mapping[str, Any],
    label: str,
    report_dir: Path,
    out_dir: Path,
    all_lane_c: Sequence[Tactic] | None = None,
    without_cameras: str | None = None,
) -> str:
    """Both candidates (the main recommendation and, when given, the camera-free one) go
    through their own gate; each that passes is added. Writes the report, or SKIPPED.md when
    nothing was added. Returns one line saying which."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, str, Sequence[str]]] = [(REC_ROLE, label, MAIN_STAGES)]
    if without_cameras is not None and without_cameras != label:
        candidates.append((NOCAMS_ROLE, without_cameras, (NOCAMS,)))
    report_path, detail_path = report_dir / "report.json", report_dir / "report_detail.json"
    have_report = report_path.is_file() and detail_path.is_file()
    report: Report | None = None
    detail: dict[str, Any] = {}
    problem: str | None = None
    if not have_report:
        problem = f"{report_path} or {detail_path} does not exist, so there is nothing to add to"
    else:
        report = Report.model_validate_json(report_path.read_text())
        detail = dict(json.loads(detail_path.read_text()))
        if report.baseline_config != ctx.baseline.name:
            problem = (
                f"the report's baseline is {report.baseline_config!r}, the campaign's is "
                f"{ctx.baseline.name!r}"
            )

    added: dict[str, dict[str, Any]] = {}
    not_added: dict[str, dict[str, Any]] = {}
    for role, candidate, names in candidates:
        verdict = gate(stages, candidate, names)
        reason: str | None = None
        if not verdict["passed"]:
            reason = f"{verdict['reason']}: interval {verdict['worst_delta_ci']}"
        elif problem is not None or report is None:
            reason = problem
        else:
            try:
                taken = [c.config_name for c in report.configs]
                result, config_detail, extra = config_for_report(
                    ctx, stages, candidate, verdict, taken, all_lane_c
                )
            except (HandoffError, KeyError, ValueError) as err:
                reason = f"it could not be added: {err}"
            else:
                report = Report.model_validate(
                    report.model_copy(update={"configs": [*report.configs, result]}).model_dump()
                )
                configs = {
                    **detail.get("configs", {}),
                    result.config_name: config_detail.model_dump(),
                }
                detail["configs"] = configs
                added[role] = extra
        if reason is not None:
            not_added[role] = {"label": candidate, "reason": reason, "gate": verdict}

    for name in ("report.json", "report_detail.json", "README.md", "SKIPPED.md"):
        (out_dir / name).unlink(missing_ok=True)
    criterion = (
        "Criterion (the same as fix.verdict_for), applied to each candidate in its own stage: "
        "the lower end of the paired interval for worst-tactic detection against the baseline, "
        "on final seeds, must be above zero."
    )
    refused = [
        f"- {role}, `{row['label']}`: {row['reason']} (stage {row['gate']['stage']}, interval "
        f"{row['gate']['worst_delta_ci']})"
        for role, row in not_added.items()
    ]
    if not added or report is None:
        text = "\n".join(["# Contract report not rebuilt", "", criterion, "", *refused, ""])
        (out_dir / "SKIPPED.md").write_text(text)
        return "contract report skipped: " + "; ".join(
            f"{role}: {row['reason']}" for role, row in not_added.items()
        )
    detail["campaign_recommended"] = added
    detail["campaign_not_added"] = not_added
    (out_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    Report.model_validate_json((out_dir / "report.json").read_text())
    _write_json(out_dir / "report_detail.json", detail)
    readme = [
        "# Contract report with the campaign's recommended configurations",
        "",
        f"`report.json` is `{report_path}` plus {len(added)} configuration(s). "
        "`report_detail.json` is its sidecar. The originals are untouched.",
        "",
        criterion,
        "",
    ]
    for role, extra in added.items():
        readme += [
            f"## Added: {role}, `{extra['config_name']}` (campaign label `{extra['label']}`)",
            "",
            f"Its interval in {extra['stage']} is {extra['gate']['worst_delta_ci']}, which "
            "excludes zero.",
            "",
            "What the contract cannot say about this row:",
            "",
            *(f"- {line}" for line in extra["what_the_contract_cannot_say"]),
            "",
        ]
    if refused:
        readme += ["## Not added", "", *refused, ""]
    readme += [
        "The same, with the numbers, is under `campaign_recommended` and `campaign_not_added` "
        "in the sidecar.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme))
    return f"contract report written to {out_dir}: added {sorted(added)}" + (
        f", not added {sorted(not_added)}" if not_added else ""
    )


# ---- driver ----------------------------------------------------------------------------------


def load_ctx(
    out: Path,
    results: Mapping[str, Any],
    ev: Evaluator,
    curves: SensorCurves,
    scenario_dir: Path,
    tactic_dirs: Sequence[Path],
    smoke: bool,
    campaign_root: Path,
) -> tuple[Ctx, list[Tactic]]:
    """A read-only campaign context: nothing here writes under out. Also returns every lane C
    tactic on disk, because a smoke run keeps only the first few."""
    site = Site.model_validate_json((scenario_dir / "site.json").read_text())
    spec = json.loads((scenario_dir / "fleets" / "sweep.json").read_text())
    baseline = FleetConfig.model_validate_json(
        (scenario_dir / "fleets" / f"{spec['baseline']}.json").read_text()
    )
    limits = adversary.load_limits(scenario_dir)
    lane_c, _ = adversary.load_tactic_files([p for p in tactic_dirs if p.exists()], site, limits)
    sizes = campaign.SMOKE if smoke else campaign.FULL
    kept = lane_c[: sizes.max_lane_c] if sizes.max_lane_c is not None else lane_c
    ctx = campaign.Ctx(
        out=out,
        sizes=sizes,
        site=site,
        curves=curves,
        baseline=baseline,
        limits=limits,
        costs={str(k): float(v) for k, v in results.get("costs", {}).items()},
        seeds=seedsplit.load(campaign_root / "seeds.json"),
        ev=ev,
        lane_c_tactics=kept,
        started=0.0,
        deadline=0.0,
        checkpoint={},
    )
    return ctx, lane_c


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the campaign's output directory")
    parser.add_argument("--handoff-dir", type=Path, default=None, help="write here, not in --out")
    parser.add_argument("--note-dir", type=Path, default=DEFAULT_NOTE_DIR)
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument("--tactics-dir", type=Path, action="append", default=None)
    parser.add_argument("--report-dir", type=Path, default=None, help="holds report.json")
    parser.add_argument("--sizes", choices=("auto", "full", "smoke"), default="auto")
    args = parser.parse_args(argv)
    if not args.out.is_absolute():
        parser.error("--out must be an absolute path")
    # every Config below is built through official_params(), which reads the environment:
    # a stray policy variable would silently point the handoff at other cache files
    with policy_env(None, None):
        return _handoff(args)


def _handoff(args: argparse.Namespace) -> int:
    out: Path = args.out
    dest: Path = args.handoff_dir or out
    nested = out.name == "smoke" and not (out / "seeds.json").is_file()
    campaign_root = out.parent if nested else out
    smoke = nested if args.sizes == "auto" else args.sizes == "smoke"
    data_dir = campaign_root.parent
    scenario_dir = args.scenario_dir or data_dir / "lane_c_export" / "scenarios" / "logistics_yard"
    tactic_dirs = args.tactics_dir or [data_dir / "lane_c_export" / "results"]
    report_dir = args.report_dir or data_dir

    def stop(reason: str) -> int:
        print(f"handoff: {reason}")
        _write_json(dest / "replays" / "index.json", {"complete": False, "reason": reason})
        return 0

    results_path = out / "results.json"
    if not results_path.is_file():
        return stop(f"{results_path} does not exist: the campaign has not finished")
    if not (out / "cache").is_dir():
        return stop(f"{out / 'cache'} does not exist")
    results = json.loads(results_path.read_text())
    stages: dict[str, Any] = results.get("stages", {})
    label = recommended_label(stages)
    if label is None:
        return stop("the validation stage is not complete, so no configuration is recommended")

    curves = SensorCurves.model_validate_json((scenario_dir / "sensor_curve.json").read_text())
    with Evaluator(out / "cache", curves, None, 1) as ev:
        ctx, all_lane_c = load_ctx(
            out, results, ev, curves, scenario_dir, tactic_dirs, smoke, campaign_root
        )
        chosen = {e.label: e for e in campaign.chosen_entries(ctx, stages["validation"])}
        written: dict[str, dict[str, Any]] = {}
        for folder, wanted in (
            ("recommended", label),
            ("best_free_policy", campaign.BEST_FREE_LABEL),
        ):
            if wanted in chosen:
                written[folder] = write_config_files(chosen[wanted], ctx.site, dest / folder)
                print(f"handoff: {folder} ({wanted}) written to {dest / folder}")
            else:
                print(f"handoff: no configuration is labelled {wanted!r}; {folder}/ not written")
        if "recommended" not in written:
            return stop(f"the recommended configuration {label!r} is not among the chosen ones")

        def replays(role: str, wanted: str, names: tuple[str, ...], folder: str) -> None:
            try:
                index = export_replays(
                    ctx, stages, wanted, written[role], dest / folder, all_lane_c, names
                )
            except HandoffError as err:
                index = {"complete": False, "reason": str(err), "recommended_label": wanted}
            _write_json(dest / folder / "index.json", index)
            print(
                f"handoff: {folder}: {index.get('pairs_exported', 0)} pairs"
                + ("" if index["complete"] else f" ({index['reason']})")
            )

        replays(REC_ROLE, label, MAIN_STAGES, "replays")

        without = nocams_label(stages)
        nocams_dir, nocams_replays = dest / NOCAMS_ROLE, dest / "replays_without_cameras"
        if without is None:
            why = f"the optional stage {NOCAMS} is missing or incomplete"
            print(f"handoff: {NOCAMS_ROLE}/ not written: {why}")
            _write_json(nocams_replays / "index.json", {"complete": False, "reason": why})
        else:
            try:
                entry = entries_of(ctx, stages[NOCAMS])[without]
                written[NOCAMS_ROLE] = write_config_files(entry, ctx.site, nocams_dir)
                print(f"handoff: {NOCAMS_ROLE} ({without}) written to {nocams_dir}")
            except (HandoffError, KeyError) as err:
                why = f"{NOCAMS_ROLE} could not be rebuilt from {NOCAMS}: {err}"
                print(f"handoff: {why}")
                _write_json(nocams_replays / "index.json", {"complete": False, "reason": why})
                without = None
            else:
                replays(NOCAMS_ROLE, without, (NOCAMS,), "replays_without_cameras")

        args.note_dir.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[3]
        note = lane_c_note(written, campaign_root, scenario_dir, repo_root)
        (args.note_dir / NOTE_NAME).write_text(note)
        print(f"handoff: note for lane C written to {args.note_dir / NOTE_NAME}")

        print(
            "handoff: "
            + write_contract_report(
                ctx, stages, label, report_dir, dest / "contract_report", all_lane_c, without
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
