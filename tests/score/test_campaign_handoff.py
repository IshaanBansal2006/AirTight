from __future__ import annotations

import dataclasses
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from airtight.contracts import Conditions, ConfigResult, PairedDelta, Report
from airtight.score import adversary, campaign, hardware, seedsplit
from airtight.score import campaign_handoff as handoff
from airtight.score.cells import Evaluator
from airtight.score.policy import named_policy, sample_policy
from airtight.score.report import METRIC_WORST
from airtight.sim import scenarios
from airtight.sim.constants import NEVER_SEEN
from airtight.sim.episode import PARAMS_JSON_ENV, official_params

if TYPE_CHECKING:
    from airtight.score.cells import Config

import numpy as np

LIMITS = {
    "speed_min_mps": 0.8,
    "speed_cap_mps": 2.0,
    "max_waypoints": 6,
    "perimeter_tol_m": 1.5,
    "asset_tol_m": 1.0,
    "min_leg_m": 2.0,
}
TOY = dataclasses.replace(
    campaign.SMOKE,
    specs=("d2_std_nocams", "d3_std_nocams"),
    final_seeds=(3, 3),
    final_quiet=1,
    standin_phases=(1, 1),
    n_boot=20,
)
REC = "best:d3_std_nocams"
WATCHED = (handoff.ENGINE_ENV, PARAMS_JSON_ENV, handoff.WEIGHT_MODE_ENV, handoff.TASK_TIME_ENV)


@dataclasses.dataclass
class Toy:
    out: Path
    scenario_dir: Path
    ctx: campaign.Ctx
    stages: dict[str, Any]
    nocams: dict[str, Any]

    def argv(self, tmp: Path, *more: str) -> list[str]:
        return [
            "--out",
            str(self.out),
            "--handoff-dir",
            str(tmp / "handoff"),
            "--note-dir",
            str(tmp / "note"),
            "--scenario-dir",
            str(self.scenario_dir),
            "--tactics-dir",
            str(tmp / "no_tactics"),
            "--report-dir",
            str(tmp / "reports"),
            *more,
        ]


def _scenario(root: Path) -> Path:
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    fleet = scenarios.load_fleet("1drone")
    (root / "fleets").mkdir(parents=True)
    (root / "site.json").write_text(site.model_dump_json())
    (root / "sensor_curve.json").write_text(curves.model_dump_json())
    (root / "fleets" / "sweep.json").write_text(json.dumps({"baseline": "base"}))
    (root / "fleets" / "base.json").write_text(fleet.model_dump_json())
    (root / "redteam_config.json").write_text(json.dumps(LIMITS))
    return root


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Toy:
    """A finished toy campaign on yard_night: a validation record and a real final_standin
    stage, a few dozen cheap episodes, written the way campaign.main writes them."""
    tmp = tmp_path_factory.mktemp("campaign")
    scenario_dir = _scenario(tmp / "data" / "scenario")
    root = tmp / "data" / "campaign"
    out = root / "smoke"
    seedsplit.write_seeds(root / "seeds.json")
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    baseline = scenarios.load_fleet("1drone")
    costs = {"entry_camera_usd_per_hour": 0.4, "swap_dock_per_drone_usd_per_hour": 1.26}
    split = seedsplit.load(root / "seeds.json")
    with Evaluator(out / "cache", curves, split, 1) as ev:
        ctx = campaign.Ctx(
            out=out,
            sizes=TOY,
            site=site,
            curves=curves,
            baseline=baseline,
            limits=adversary.load_limits(scenario_dir),
            costs=costs,
            seeds=split,
            ev=ev,
            lane_c_tactics=[],
            started=time.time(),
            deadline=time.time() + 3600.0,
            checkpoint={"rate": 1.0},
        )
        chosen = {}
        for spec in campaign.specs_for(ctx):
            fleet = hardware.build_fleet(baseline, spec)
            if spec.name == "d2_std_nocams":
                policy = named_policy(fleet, "asset")
            else:
                policy = sample_policy(site, fleet, np.random.default_rng([7, 7]))
            entry = campaign.policy_entry(ctx, spec, policy, spec.name)
            tactics = campaign._final_set(ctx, [entry])
            chosen[spec.name] = {
                "hardware": spec.name,
                "policy": policy.describe(),
                "worst_naive": {"tactic": tactics[0].id},
            }
        validation = {
            "complete": True,
            "chosen": chosen,
            "recommended": "d3_std_nocams",
            "frontier_labels": [REC],
        }
        stages = {
            "validation": validation,
            "final_standin": campaign.stage_final_standin(ctx, validation),
        }
        free = [
            e
            for e in campaign.chosen_entries(ctx, validation)
            if e.label in (campaign.BASELINE_LABEL, campaign.BEST_FREE_LABEL)
        ]
        nocams = campaign.strong_evaluate(ctx, free, handoff.NOCAMS)
        nocams["recommended_without_cameras"] = "d2_std_nocams"
        nocams["recommended_without_cameras_label"] = campaign.BEST_FREE_LABEL
    results = {"stages": stages, "costs": costs, "checkpoint": {}}
    (out / "results.json").write_text(json.dumps(results, sort_keys=True))
    ctx.ev = Evaluator(out / "cache", curves, None, 1)
    plain = json.loads(json.dumps({"stages": stages, "nocams": nocams}, sort_keys=True))
    return Toy(out, scenario_dir, ctx, plain["stages"], plain["nocams"])


class FakeRows:
    def __init__(self, rows: dict[tuple[str, int], dict[str, Any]]) -> None:
        self.rows = rows

    def row(self, config: Config, kind: str, seed: int) -> dict[str, Any] | None:
        return self.rows.get((config.name, seed))


def _row(peak: float, t_alarm: float | None) -> dict[str, Any]:
    return {"intruder_peak": peak, "intruder_t_alarm_ref": t_alarm, "t_cdp": 50.0}


def test_find_pairs_reads_each_side_at_its_own_threshold_in_seed_order(toy: Toy) -> None:
    entries = handoff.entries_of(toy.ctx, toy.stages["final_standin"])
    base, rec = entries[campaign.BASELINE_LABEL].config, entries[REC].config
    tactic = scenarios.load_tactic("jog")
    rows = {
        (base.name, 1): _row(1.0, None),  # miss, and a miss in the log
        (rec.name, 1): _row(3.5, 60.0),  # catch at its tau 3, late at TAU_REF: not in the log
        (base.name, 2): _row(NEVER_SEEN, None),
        (rec.name, 2): _row(9.0, 10.0),  # a pair in the log too
        (base.name, 3): _row(2.5, None),  # the baseline catches at its tau 2: no pair
        (rec.name, 3): _row(9.0, 10.0),
        (base.name, 4): _row(1.0, None),
        (rec.name, 4): _row(NEVER_SEEN, None),  # never seen is never a catch
        (base.name, 5): _row(1.0, None),  # seed 5: the recommended row is not cached
        (base.name, 6): _row(1.9, 80.0),
        (rec.name, 6): _row(3.0, 50.0),  # at the threshold counts, and on time at t_cdp
    }
    pairs, examined = handoff.find_pairs(
        FakeRows(rows), base, rec, tactic, [6, 1, 2, 3, 4, 5], 2.0, 3.0
    )
    assert examined == 5
    assert [p.seed for p in pairs] == [6, 1, 2]
    assert [p.also_in_log for p in pairs] == [True, False, True]
    assert [p.seed for p in handoff.order_pairs(pairs, 2)] == [6, 2]
    assert [p.seed for p in handoff.order_pairs(pairs)] == [6, 2, 1]


def _stages(ci: list[float], stage: str = "final_strong") -> dict[str, Any]:
    delta = {"a": REC, "b": "baseline", "worst_delta": 0.2, "worst_delta_ci": ci, "n_seeds": 9}
    return {stage: {"complete": True, "paired_vs_baseline": [delta]}}


def test_gate_needs_the_interval_above_zero_and_prefers_the_strong_stage() -> None:
    assert handoff.gate(_stages([0.05, 0.4]), REC)["passed"] is True
    for ci in ([0.0, 0.4], [-0.1, 0.4]):
        verdict = handoff.gate(_stages(ci), REC)
        assert verdict["passed"] is False and verdict["worst_delta_ci"] == ci
    both = {**_stages([-0.1, 0.4]), **_stages([0.1, 0.4], "final_standin")}
    assert handoff.gate(both, REC)["stage"] == "final_strong"
    assert handoff.gate(both, REC)["passed"] is False
    other = {**_stages([0.1, 0.4], "final_standin"), **_stages([0.1, 0.4])}
    other["final_strong"]["paired_vs_baseline"][0]["a"] = "someone else"
    assert handoff.gate(other, REC)["stage"] == "final_standin"
    incomplete = _stages([0.1, 0.4])
    incomplete["final_strong"]["complete"] = False
    assert handoff.gate(incomplete, REC)["passed"] is False
    assert handoff.gate({}, REC)["stage"] is None


def test_pick_stage_and_the_baselines_worst_tactic() -> None:
    row = {"label": "baseline", "worst_naive": {"tactic": "n"}}
    rec = {"label": REC, "worst_naive": {"tactic": "x"}}
    strong = {"complete": True, "final": [row]}
    standin = {"complete": True, "rows": [row, rec]}
    picked = handoff.pick_stage(
        {"final_strong": strong, "final_standin": standin}, ["baseline", REC]
    )
    assert picked is not None and picked[0] == "final_standin"
    strong["final"].append(rec)
    picked = handoff.pick_stage(
        {"final_strong": strong, "final_standin": standin}, ["baseline", REC]
    )
    assert picked is not None and picked[0] == "final_strong"
    assert (
        handoff.pick_stage({"final_standin": {"complete": False, "rows": [row, rec]}}, [REC])
        is None
    )
    assert handoff.baseline_worst_tactic(row) == ("n", "worst_naive")
    held = {**row, "worst_heldout": {"tactic": "h"}}
    assert handoff.baseline_worst_tactic(held) == ("h", "worst_heldout")


def test_recommended_label() -> None:
    assert handoff.recommended_label({}) is None
    assert handoff.recommended_label({"validation": {"complete": False}}) is None
    done = {"validation": {"complete": True, "recommended": "d2_std_nocams"}}
    assert handoff.recommended_label(done) == campaign.BEST_FREE_LABEL
    done["validation"]["recommended"] = "d3_swap_cams"
    assert handoff.recommended_label(done) == "best:d3_swap_cams"


def test_params_json_round_trips_through_official_params(
    toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(handoff.WEIGHT_MODE_ENV, "band")  # must not leak into the check
    monkeypatch.delenv(PARAMS_JSON_ENV, raising=False)
    entry = handoff.entries_of(toy.ctx, toy.stages["final_standin"])[REC]
    written = handoff.write_config_files(entry, toy.ctx.site, tmp_path / "recommended")
    assert os.environ[handoff.WEIGHT_MODE_ENV] == "band" and PARAMS_JSON_ENV not in os.environ
    assert written["site_variant"] is None and not (tmp_path / "recommended" / "site.json").exists()
    assert written["env"] == {"AIRTIGHT_ENGINE": "v0", PARAMS_JSON_ENV: written["params_json"]}
    monkeypatch.delenv(handoff.WEIGHT_MODE_ENV)
    monkeypatch.setenv(PARAMS_JSON_ENV, written["params_json"])
    assert official_params() == entry.config.params
    monkeypatch.delenv(PARAMS_JSON_ENV)
    assert official_params() != entry.config.params
    fleet = json.loads(Path(written["fleet"]).read_text())
    assert fleet["charge_policy"]["stagger_offsets_s"] == {
        a: v
        for a, v in entry.policy["offsets"].items()
        if v > 0  # type: ignore[index, union-attr]
    }
    params_path = Path(written["params_json"])
    params_path.write_text(json.dumps({"d0_m": 1.0}))
    with pytest.raises(handoff.HandoffError, match="does not read back"):
        handoff.verify_params_file(params_path, entry.config.params)


def test_a_site_variant_is_written_only_when_the_site_differs(toy: Toy, tmp_path: Path) -> None:
    entry = handoff.entries_of(toy.ctx, toy.stages["final_standin"])[REC]
    other = toy.ctx.site.model_copy(update={"name": "somewhere else"})
    written = handoff.write_config_files(entry, other, tmp_path)
    assert written["site_variant"] == str((tmp_path / "site.json").resolve())


def test_rebuilt_entries_and_tactics_match_the_stage_or_are_refused(toy: Toy) -> None:
    stage = toy.stages["final_standin"]
    entries = handoff.entries_of(toy.ctx, stage)
    wanted = set(stage["rows"][0]["pd_by_tactic"])
    tactics = handoff.stage_tactics(toy.ctx, "final_standin", list(entries.values()), wanted)
    assert {t.id for t in tactics} == wanted
    with pytest.raises(handoff.HandoffError, match="does not match"):
        handoff.stage_tactics(toy.ctx, "final_standin", list(entries.values()), wanted | {"x"})
    broken = json.loads(json.dumps(stage))
    broken["entries"][0]["fleet_hash"] = "0" * 12
    with pytest.raises(handoff.HandoffError, match="fleet_hash"):
        handoff.entries_of(toy.ctx, broken)


def test_main_writes_everything_and_restores_the_environment(
    toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "SMOKE", TOY)
    monkeypatch.setenv(handoff.ENGINE_ENV, "stub")
    monkeypatch.setenv(handoff.WEIGHT_MODE_ENV, "band")
    monkeypatch.delenv(PARAMS_JSON_ENV, raising=False)
    monkeypatch.delenv(handoff.TASK_TIME_ENV, raising=False)
    before = {name: os.environ.get(name) for name in WATCHED}

    real_find = handoff.find_pairs

    def forced(*args: Any) -> tuple[list[handoff.Pair], int]:
        pairs, examined = real_find(*args)
        seeds = args[4]
        return pairs or [handoff.Pair(seeds[0], 0.0, 9.0, False, True)], examined

    seen: list[dict[str, str | None]] = []
    real_run = handoff.run_episode

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append({name: os.environ.get(name) for name in WATCHED})
        return real_run(*args, **kwargs)

    monkeypatch.setattr(handoff, "find_pairs", forced)
    monkeypatch.setattr(handoff, "run_episode", spy)
    cache_before = sorted(
        (p, p.stat().st_size) for p in (toy.out / "cache").rglob("*") if p.is_file()
    )

    assert handoff.main(toy.argv(tmp_path)) == 0

    assert {name: os.environ.get(name) for name in WATCHED} == before
    dest = tmp_path / "handoff"
    params = str((dest / "recommended" / "params.json").resolve())
    assert seen and len(seen) % 2 == 0
    for baseline_env, recommended_env in zip(seen[::2], seen[1::2], strict=True):
        assert baseline_env == {**dict.fromkeys(WATCHED), handoff.ENGINE_ENV: "v0"}
        assert recommended_env == {**baseline_env, PARAMS_JSON_ENV: params}
    index = json.loads((dest / "replays" / "index.json").read_text())
    assert index["complete"] and index["stage"] == "final_standin"
    assert 1 <= index["pairs_exported"] <= handoff.MAX_PAIRS
    base_row = next(r for r in toy.stages["final_standin"]["rows"] if r["label"] == "baseline")
    assert index["tactic"]["id"] == base_row["worst_naive"]["tactic"]
    for pair in index["pairs"]:
        for side in ("baseline", "recommended"):
            row = pair[side]
            assert Path(row["log_path"]).parent == dest / "replays" / side
            assert row["plan_replay"]["ok"] in (True, None), row["plan_replay"]
            if row["plan_replay"]["ok"]:
                assert row["plan_replay"]["seed"] == pair["seed"]
                logged = row["timely_detected_in_log_at_tau_ref"]
                assert row["plan_replay"]["timely_detected"] == logged
            assert row["log_agrees_with_cache"] is True
        assert pair["miss_then_catch_in_the_log_itself"] == (
            pair["recommended"]["timely_detected_in_log_at_tau_ref"]
            and not pair["baseline"]["timely_detected_in_log_at_tau_ref"]
        )
    assert (dest / "best_free_policy" / "fleet.json").is_file()
    note = (tmp_path / "note" / handoff.NOTE_NAME).read_text()
    assert params in note and "ONLY when" in note and "attacks_on_recommended" in note
    commands = note.split("```")[1::2]
    assert commands and all("#" not in block for block in commands)
    skipped = (dest / "contract_report" / "SKIPPED.md").read_text()
    assert "report.json" in skipped or "does not exclude zero" in skipped
    nocams = json.loads((dest / "replays_without_cameras" / "index.json").read_text())
    assert nocams["complete"] is False and handoff.NOCAMS in nocams["reason"]
    assert not (dest / handoff.NOCAMS_ROLE).exists()
    assert "No camera-free recommendation is on record" in note
    cache_after = sorted(
        (p, p.stat().st_size) for p in (toy.out / "cache").rglob("*") if p.is_file()
    )
    assert cache_after == cache_before
    assert sorted(p.name for p in toy.out.iterdir()) == ["cache", "checkpoint.json", "results.json"]


def _toy_report(toy: Toy, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    base = ConfigResult(
        config_name=toy.ctx.baseline.name,
        fleet_hash=toy.ctx.baseline.content_hash(),
        n_episodes=1,
        roc=[],
        pd_at_operating_point=0.5,
        pd_at_operating_point_ci=(0.4, 0.6),
        worst_tactic_id="t",
        worst_tactic_pd=0.1,
        cost_per_hour=1.0,
        coverage_gap_s_per_hour=0.0,
        human_decisions_per_hour=0.0,
    )
    report = Report(
        generated_at=datetime(2026, 9, 19, tzinfo=UTC),
        site_hash=toy.ctx.site.content_hash(),
        baseline_config=base.config_name,
        conditions=Conditions(
            far_per_hour_operating_point=1.0,
            adversary_knowledge="a",
            sensor_calibration="s",
            detection_model_note="d",
            seed_list_hash="h",
            n_seeds=1,
        ),
        configs=[base],
    )
    (directory / "report.json").write_text(report.model_dump_json())
    (directory / "report_detail.json").write_text(json.dumps({"configs": {}, "fix": {"kept": 1}}))


def _with_interval(toy: Toy, ci: list[float]) -> dict[str, Any]:
    stages = json.loads(json.dumps(toy.stages))
    for delta in stages["final_standin"]["paired_vs_baseline"]:
        if delta["a"] == REC:
            delta["worst_delta_ci"] = ci
    return stages


def test_contract_report_is_skipped_when_the_interval_includes_zero(
    toy: Toy, tmp_path: Path
) -> None:
    _toy_report(toy, tmp_path / "reports")
    out_dir = tmp_path / "contract_report"
    out_dir.mkdir()
    (out_dir / "report.json").write_text("stale")
    line = handoff.write_contract_report(
        toy.ctx, _with_interval(toy, [-0.2, 0.3]), REC, tmp_path / "reports", out_dir
    )
    assert "skipped" in line
    assert sorted(p.name for p in out_dir.iterdir()) == ["SKIPPED.md"]
    assert "[-0.2, 0.3]" in (out_dir / "SKIPPED.md").read_text()


def test_contract_report_is_skipped_without_an_existing_report(toy: Toy, tmp_path: Path) -> None:
    line = handoff.write_contract_report(
        toy.ctx, _with_interval(toy, [0.2, 0.3]), REC, tmp_path / "nowhere", tmp_path / "cr"
    )
    assert "does not exist" in line and (tmp_path / "cr" / "SKIPPED.md").is_file()


def test_contract_report_refuses_an_interval_the_cache_does_not_reproduce(
    toy: Toy, tmp_path: Path
) -> None:
    _toy_report(toy, tmp_path / "reports")
    verdict = handoff.gate(toy.stages, REC)
    if verdict["passed"]:
        pytest.skip("the toy campaign happens to pass the gate")
    line = handoff.write_contract_report(
        toy.ctx, _with_interval(toy, [0.2, 0.3]), REC, tmp_path / "reports", tmp_path / "cr"
    )
    assert "does not exclude zero" in line


def test_contract_report_adds_the_recommended_configuration_when_the_gate_passes(
    toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_report(toy, tmp_path / "reports")
    original = (tmp_path / "reports" / "report.json").read_text()
    real = handoff.paired_deltas

    def lifted(*args: Any, **kwargs: Any) -> list[PairedDelta]:
        return [
            d.model_copy(update={"ci": (0.2, 0.3)}) if d.metric == METRIC_WORST else d
            for d in real(*args, **kwargs)
        ]

    monkeypatch.setattr(handoff, "paired_deltas", lifted)
    out_dir = tmp_path / "contract_report"
    line = handoff.write_contract_report(
        toy.ctx, _with_interval(toy, [0.2, 0.3]), REC, tmp_path / "reports", out_dir
    )
    assert "written" in line, line
    assert (tmp_path / "reports" / "report.json").read_text() == original
    report = Report.model_validate_json((out_dir / "report.json").read_text())
    entry = handoff.entries_of(toy.ctx, toy.stages["final_standin"])[REC]
    added = report.config(entry.config.name)
    rec_row = next(r for r in toy.stages["final_standin"]["rows"] if r["label"] == REC)
    assert added.pd_at_operating_point == rec_row["pd"]
    assert added.n_episodes == rec_row["n_seeds"] * rec_row["n_tactics"]
    assert added.cost_per_hour == entry.cost_per_hour
    assert {d.metric for d in added.paired_vs_baseline} >= {METRIC_WORST}
    detail = json.loads((out_dir / "report_detail.json").read_text())
    assert detail["fix"] == {"kept": 1} and entry.config.name in detail["configs"]
    assert detail["campaign_not_added"] == {}
    extra = detail["campaign_recommended"][handoff.REC_ROLE]
    assert extra["recomputed_from_cache_equals_results_json"]["tau"] is True
    assert extra["recomputed_from_cache_equals_results_json"]["pd"] is True
    assert any("AIRTIGHT_PARAMS_JSON" in line for line in extra["what_the_contract_cannot_say"])
    assert "cannot say" in (out_dir / "README.md").read_text()
    assert not (out_dir / "SKIPPED.md").exists()


@pytest.mark.parametrize("stages", [{}, {"validation": {"complete": False}}])
def test_missing_stages_exit_cleanly_with_an_explanation(
    toy: Toy, tmp_path: Path, stages: dict[str, Any]
) -> None:
    out = tmp_path / "campaign"
    (out / "cache").mkdir(parents=True)
    (out / "results.json").write_text(json.dumps({"stages": stages}))
    argv = toy.argv(tmp_path)
    argv[1] = str(out)
    assert handoff.main(argv) == 0
    index = json.loads((tmp_path / "handoff" / "replays" / "index.json").read_text())
    assert index["complete"] is False and "validation" in index["reason"]


def test_no_results_file_and_no_final_stage_are_both_explained(
    toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = toy.argv(tmp_path)
    argv[1] = str(tmp_path / "empty")
    assert handoff.main(argv) == 0
    index = json.loads((tmp_path / "handoff" / "replays" / "index.json").read_text())
    assert "has not finished" in index["reason"]

    monkeypatch.setattr(campaign, "SMOKE", TOY)
    cut = tmp_path / "data" / "campaign" / "smoke"
    cut.mkdir(parents=True)
    (cut / "cache").mkdir()
    (cut.parent / "seeds.json").write_text((toy.out.parent / "seeds.json").read_text())
    results = {"stages": {"validation": toy.stages["validation"]}, "costs": toy.ctx.costs}
    (cut / "results.json").write_text(json.dumps(results))
    argv[1] = str(cut)
    assert handoff.main(argv) == 0
    index = json.loads((tmp_path / "handoff" / "replays" / "index.json").read_text())
    assert index["complete"] is False and "final_strong" in index["reason"]
    assert (tmp_path / "handoff" / "recommended" / "params.json").is_file()
    assert (tmp_path / "handoff" / "contract_report" / "SKIPPED.md").is_file()


def test_the_smoke_output_when_present(tmp_path: Path) -> None:
    smoke = Path(__file__).parents[2] / "data" / "campaign" / "smoke"
    if not (smoke / "results.json").is_file():
        pytest.skip("no smoke campaign output on this machine")
    results = json.loads((smoke / "results.json").read_text())
    label = handoff.recommended_label(results["stages"])
    assert label is not None
    assert handoff.pick_stage(results["stages"], [campaign.BASELINE_LABEL, label]) is not None
    assert handoff.gate(results["stages"], label)["stage"] in ("final_strong", "final_standin")


def _with_nocams(toy: Toy, ci: list[float] | None = None) -> dict[str, Any]:
    """The toy stages plus the real final_strong_nocams stage, its paired interval replaced
    when ci is given."""
    stages = json.loads(json.dumps(toy.stages))
    stages[handoff.NOCAMS] = json.loads(json.dumps(toy.nocams))
    for delta in stages[handoff.NOCAMS]["paired_vs_baseline"]:
        if ci is not None:
            delta["worst_delta_ci"] = ci
    return stages


def test_nocams_label_needs_a_complete_stage(toy: Toy) -> None:
    assert handoff.nocams_label({}) is None
    assert handoff.nocams_label(toy.stages) is None
    stages = _with_nocams(toy)
    assert handoff.nocams_label(stages) == "best_free_policy"
    stages[handoff.NOCAMS]["complete"] = False
    assert handoff.nocams_label(stages) is None
    assert handoff.gate(stages, "best_free_policy", (handoff.NOCAMS,))["stage"] is None


def test_gate_reads_each_candidate_in_its_own_stage(toy: Toy) -> None:
    stages = _with_nocams(_as_toy(toy, _with_interval(toy, [-0.1, 0.2])), [0.3, 0.4])
    assert handoff.gate(stages, REC)["passed"] is False
    verdict = handoff.gate(stages, "best_free_policy", (handoff.NOCAMS,))
    assert verdict["passed"] is True and verdict["stage"] == handoff.NOCAMS
    # the stand-in stage also holds a row for that label: it is never the camera-free gate
    assert handoff.gate(stages, "best_free_policy")["stage"] == "final_standin"


def _as_toy(toy: Toy, stages: dict[str, Any]) -> Toy:
    return dataclasses.replace(toy, stages=stages)


def test_skipped_lists_both_intervals_when_neither_passes(toy: Toy, tmp_path: Path) -> None:
    _toy_report(toy, tmp_path / "reports")
    stages = _with_nocams(_as_toy(toy, _with_interval(toy, [-0.2, 0.3])), [-0.4, 0.1])
    line = handoff.write_contract_report(
        toy.ctx, stages, REC, tmp_path / "reports", tmp_path / "cr", None, "best_free_policy"
    )
    assert "skipped" in line
    text = (tmp_path / "cr" / "SKIPPED.md").read_text()
    assert "[-0.2, 0.3]" in text and "[-0.4, 0.1]" in text
    assert handoff.REC_ROLE in text and handoff.NOCAMS_ROLE in text


def test_contract_report_adds_each_candidate_that_passes(
    toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_report(toy, tmp_path / "reports")
    real = handoff.paired_deltas

    def lifted(*args: Any, **kwargs: Any) -> list[PairedDelta]:
        return [
            d.model_copy(update={"ci": (0.2, 0.3)}) if d.metric == METRIC_WORST else d
            for d in real(*args, **kwargs)
        ]

    monkeypatch.setattr(handoff, "paired_deltas", lifted)
    entries = handoff.entries_of(toy.ctx, toy.stages["final_standin"])
    rec_name, free_name = entries[REC].config.name, entries["best_free_policy"].config.name

    def names(main_ci: list[float], nocams_ci: list[float], out_dir: Path) -> list[str]:
        stages = _with_nocams(_as_toy(toy, _with_interval(toy, main_ci)), nocams_ci)
        handoff.write_contract_report(
            toy.ctx, stages, REC, tmp_path / "reports", out_dir, None, "best_free_policy"
        )
        report = Report.model_validate_json((out_dir / "report.json").read_text())
        return [c.config_name for c in report.configs][1:]

    assert names([0.2, 0.3], [0.2, 0.3], tmp_path / "both") == [rec_name, free_name]
    detail = json.loads((tmp_path / "both" / "report_detail.json").read_text())
    assert set(detail["campaign_recommended"]) == {handoff.REC_ROLE, handoff.NOCAMS_ROLE}
    assert detail["campaign_recommended"][handoff.NOCAMS_ROLE]["stage"] == handoff.NOCAMS
    assert {rec_name, free_name} <= set(detail["configs"])

    assert names([-0.1, 0.3], [0.2, 0.3], tmp_path / "one") == [free_name]
    detail = json.loads((tmp_path / "one" / "report_detail.json").read_text())
    assert list(detail["campaign_not_added"]) == [handoff.REC_ROLE]
    readme = (tmp_path / "one" / "README.md").read_text()
    assert "## Not added" in readme and "[-0.1, 0.3]" in readme and free_name in readme


def test_main_with_the_nocams_stage_writes_the_second_configuration(
    toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "SMOKE", TOY)
    out = tmp_path / "data" / "campaign" / "smoke"
    out.mkdir(parents=True)
    (out / "cache").symlink_to(toy.out / "cache")
    (out.parent / "seeds.json").write_text((toy.out.parent / "seeds.json").read_text())
    stages = _with_nocams(toy)
    (out / "results.json").write_text(json.dumps({"stages": stages, "costs": toy.ctx.costs}))

    real_find = handoff.find_pairs

    def forced(*args: Any) -> tuple[list[handoff.Pair], int]:
        pairs, examined = real_find(*args)
        return pairs or [handoff.Pair(args[4][0], 0.0, 9.0, False, True)], examined

    monkeypatch.setattr(handoff, "find_pairs", forced)
    argv = toy.argv(tmp_path)
    argv[1] = str(out)
    assert handoff.main(argv) == 0
    dest = tmp_path / "handoff"
    free = handoff.entries_of(toy.ctx, toy.stages["final_standin"])["best_free_policy"]
    env = json.loads((dest / handoff.NOCAMS_ROLE / "env.json").read_text())
    assert env["config"] == free.config.name and env["site_variant"] is None
    assert not (dest / handoff.NOCAMS_ROLE / "site.json").exists()
    index = json.loads((dest / "replays_without_cameras" / "index.json").read_text())
    assert index["complete"] and index["stage"] == handoff.NOCAMS
    assert index["recommended_label"] == "best_free_policy" and index["pairs_exported"] >= 1
    for pair in index["pairs"]:
        assert pair["recommended"]["params_json"] == env["params_json"]
        assert pair["recommended"]["plan_replay"]["ok"] in (True, None)
        assert Path(pair["baseline"]["log_path"]).parent.parent == dest / "replays_without_cameras"
    main_index = json.loads((dest / "replays" / "index.json").read_text())
    assert main_index["stage"] == "final_standin" and main_index["recommended_label"] == REC
    note = (tmp_path / "note" / handoff.NOTE_NAME).read_text()
    assert note.index("WITHOUT cameras") < note.index("The main recommendation")
    assert env["params_json"] in note
    assert str((dest / "recommended" / "params.json").resolve()) in note
    commands = note.split("```")[1::2]
    assert len(commands) == 3 and all("#" not in block for block in commands)


def test_note_says_why_cameras_are_an_upper_bound_and_degrades_without_the_stage() -> None:
    def files(label: str, cameras: list[str]) -> dict[str, Any]:
        return {
            "label": label,
            "config": f"cfg-{label}",
            "hardware": "hw",
            "cost_per_hour": 1.0,
            "fleet": f"/x/{label}/fleet.json",
            "site_variant": f"/x/{label}/site.json" if cameras else None,
            "added_fixed_sensors": cameras,
            "params_json": f"/x/{label}/params.json",
        }

    paths = (Path("/d/campaign"), Path("/d/scenario"), Path("/repo"))
    both = handoff.lane_c_note(
        {handoff.REC_ROLE: files("cams", ["cam_a"]), handoff.NOCAMS_ROLE: files("plain", [])},
        *paths,
    )
    assert "UPPER BOUND" in both and "range 0" in both
    assert both.index("/x/plain/params.json") < both.index("/x/cams/params.json")
    assert "--site /x/cams/site.json" in both and "--site /d/scenario/site.json" in both
    alone = handoff.lane_c_note({handoff.REC_ROLE: files("cams", ["cam_a"])}, *paths)
    assert "No camera-free recommendation is on record" in alone and "/x/cams/fleet.json" in alone
