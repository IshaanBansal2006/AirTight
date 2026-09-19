from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from airtight.contracts import Report, read_episode_log
from airtight.score import report as report_module
from airtight.score.config_score import score_config
from airtight.score.replays import export_failures
from airtight.score.report import (
    METRIC_DECISIONS,
    METRIC_PD,
    METRIC_WORST,
    ReportDetail,
    build_report,
    write_report,
)
from airtight.score.sweep import load_inputs, run_sweep
from airtight.sim import scenarios
from airtight.sim.constants import ENGINE_IGNORES
from airtight.sim.coverage import coverage_profile, uncovered_s_per_hour

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.score.sweep import SweepResult

SEEDS = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
QUIET_SEEDS = [2000, 2001, 2002]
WHEN = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def _dump(model, path) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2))


@pytest.fixture(scope="module")
def swept(tmp_path_factory: pytest.TempPathFactory) -> tuple[SweepResult, Path]:
    tmp = tmp_path_factory.mktemp("report")
    root = tmp / "scenario"
    _dump(scenarios.load_site(), root / "site.json")
    _dump(scenarios.load_sensor_curves(), root / "sensor_curve.json")
    names = ["2drones", "2drones_staggered", "3drones_staggered"]
    for name in names:
        _dump(scenarios.load_fleet(name), root / "fleets" / f"{name}.json")
    # the same fleet under a second name: a configuration compared with itself
    _dump(scenarios.load_fleet("2drones"), root / "fleets" / "twin.json")
    spec = {"baseline": "2drones", "configs": [*names, "twin"]}
    (root / "fleets" / "sweep.json").write_text(json.dumps(spec))
    tactics = tmp / "tactics"
    for phase in (0.2, 0.458, 0.7):
        tactic = scenarios.load_tactic("jog").model_copy(
            update={"phase": phase, "id": f"jog@{phase:g}"}
        )
        _dump(tactic, tactics / f"jog_{phase:g}.json")
    inputs = load_inputs(root, [tactics])
    return run_sweep(inputs, SEEDS, QUIET_SEEDS, tmp / "sweep", workers=1), tmp


def test_report_round_trips_through_the_contract_model(swept: tuple[SweepResult, Path]) -> None:
    result, tmp = swept
    report, detail = build_report(result, n_boot=200, generated_at=WHEN)
    report_path, detail_path = write_report(report, detail, tmp / "out")
    assert Report.model_validate_json(report_path.read_text()) == report
    assert ReportDetail.model_validate_json(detail_path.read_text()) == detail
    assert [c.config_name for c in report.configs] == list(result.inputs.fleets)  # the spec's order
    assert report.baseline_config == "2drones" and report.config("2drones")
    assert list(detail.configs) == [c.config_name for c in report.configs]
    assert detail.engine_tag == result.engine_tag and detail.n_tactics == 3


def test_every_number_can_be_recomputed_from_the_cached_cells(
    swept: tuple[SweepResult, Path],
) -> None:
    result, tmp = swept
    report, detail = build_report(result, n_boot=200, generated_at=WHEN)
    reloaded = run_sweep(result.inputs, SEEDS, QUIET_SEEDS, tmp / "sweep", workers=1)
    assert reloaded.n_computed == 0  # nothing simulated: this is the cache
    again, again_detail = build_report(reloaded, n_boot=200, generated_at=WHEN)
    assert again == report and again_detail == detail

    inputs = result.inputs
    for config in report.configs:
        name, fleet = config.config_name, inputs.fleets[config.config_name]
        score = score_config(reloaded.episodes[name], reloaded.quiet[name], n_boot=200)
        assert config.pd_at_operating_point == score.pd
        assert config.pd_at_operating_point_ci == score.pd_ci
        assert (config.worst_tactic_id, config.worst_tactic_pd) == (
            score.worst_tactic_id,
            score.worst_tactic_pd,
        )
        assert config.human_decisions_per_hour == score.human_decisions_per_hour
        assert [(p.threshold, p.pd, p.pd_ci, p.far_per_hour) for p in config.roc] == [
            (r.tau, r.pd, r.pd_ci, r.far) for r in score.roc
        ]
        assert config.n_episodes == len(SEEDS) * 3
        assert (
            config.cost_per_hour == fleet.cost_per_hour()
            and config.fleet_hash == fleet.content_hash()
        )
        profile = coverage_profile(inputs.site, fleet, inputs.curves, seed=SEEDS[0])
        assert config.coverage_gap_s_per_hour == uncovered_s_per_hour(profile)
        d = detail.configs[name]
        assert (d.operating_threshold, d.flag, d.pd_by_tactic) == (
            score.tau,
            score.flag,
            score.pd_by_tactic,
        )
        assert d.raw_alerts_per_hour == score.raw_alerts_per_hour and d.quiet_hours == 3.0
    # exact, from the clocks: 2100 s of 3600 with both drones down; 600 when staggered; none by thirds
    down = {name: d.drones_down_s_per_hour for name, d in detail.configs.items()}
    assert down["2drones"] == pytest.approx(2100.0) and down["twin"] == down["2drones"]
    assert down["2drones_staggered"] == pytest.approx(600.0) and down["3drones_staggered"] == 0.0
    gaps = {c.config_name: c.coverage_gap_s_per_hour for c in report.configs}
    assert all(
        gaps[n] >= down[n] for n in down
    )  # drone-only fleets: the simulated gap adds transit
    assert gaps["3drones_staggered"] == 0.0 < gaps["2drones_staggered"] < gaps["2drones"]


def test_paired_deltas(swept: tuple[SweepResult, Path]) -> None:
    result, _ = swept
    report, _ = build_report(result, n_boot=200, generated_at=WHEN)
    assert report.config("2drones").paired_vs_baseline == []  # the baseline has none
    for config in report.configs:
        if config.config_name == "2drones":
            continue
        assert [p.metric for p in config.paired_vs_baseline] == [
            METRIC_PD,
            METRIC_WORST,
            METRIC_DECISIONS,
        ]
    # a configuration compared with itself: exactly zero, with a zero-width interval
    for delta in report.config("twin").paired_vs_baseline:
        assert delta.delta == 0.0 and delta.ci == (0.0, 0.0)
    base = report.config("2drones")
    for config in report.configs[1:3]:
        by_metric = {p.metric: p for p in config.paired_vs_baseline}
        assert by_metric[METRIC_PD].delta == pytest.approx(
            config.pd_at_operating_point - base.pd_at_operating_point
        )
        assert by_metric[METRIC_WORST].delta == pytest.approx(
            config.worst_tactic_pd - base.worst_tactic_pd
        )
        assert by_metric[METRIC_DECISIONS].delta == pytest.approx(
            config.human_decisions_per_hour - base.human_decisions_per_hour
        )
        assert all(p.ci[0] <= p.ci[1] for p in config.paired_vs_baseline)


def test_conditions_say_what_the_number_depends_on(swept: tuple[SweepResult, Path]) -> None:
    result, _ = swept
    c = build_report(result, n_boot=50, generated_at=WHEN)[0].conditions
    assert c.far_per_hour_operating_point == 1.0 and c.n_seeds == len(SEEDS)
    assert "within 15 s" in c.adversary_knowledge and "random draws" in c.adversary_knowledge
    assert (
        "stub curve" in c.sensor_calibration
        and result.inputs.curves.content_hash() in c.sensor_calibration
    )
    assert "truth association" in c.detection_model_note and "no clutter" in c.detection_model_note
    assert all(item in c.detection_model_note for item in ENGINE_IGNORES)
    assert c.seed_list_hash == report_module.seed_list_hash(SEEDS) and len(c.seed_list_hash) == 12
    assert report_module.seed_list_hash(SEEDS[:-1]) != c.seed_list_hash


def test_baseline_must_name_a_configuration(swept: tuple[SweepResult, Path]) -> None:
    import dataclasses

    result, _ = swept
    broken = dataclasses.replace(
        result, inputs=dataclasses.replace(result.inputs, baseline="ghost")
    )
    with pytest.raises(ValueError, match="'ghost'"):
        build_report(broken, n_boot=50)


def test_exported_failures_read_back_through_the_contract_reader(
    swept: tuple[SweepResult, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    result, tmp = swept
    monkeypatch.setenv("AIRTIGHT_ENGINE", "stub")  # the export must not leak its engine choice
    _, detail = build_report(result, n_boot=50, generated_at=WHEN)
    rows = export_failures(result, detail, 3, tmp / "replays")
    import os

    assert os.environ["AIRTIGHT_ENGINE"] == "stub"
    assert len(rows) == 3 and json.loads((tmp / "replays" / "index.json").read_text()) == rows
    worst = min(
        detail.configs["2drones"].pd_by_tactic, key=detail.configs["2drones"].pd_by_tactic.get
    )  # type: ignore[arg-type]
    for row in rows:
        assert row["configuration"] == "2drones" and row["tactic_id"] == worst
        assert (
            row["outcome"].startswith("not seen before") and row["intruder_peak"] is None
        )  # the charging window
        from pathlib import Path as _Path

        header, events = read_episode_log(_Path(row["log_path"]))
        kinds = [e.kind for e in events]
        assert (
            header.sim_version == "v0" and header.seed == row["seed"] and header.tactic.id == worst
        )
        assert kinds[-1] == "outcome" and "position" in kinds
    assert [r["seed"] for r in rows] == SEEDS[:3]  # ties keep seed order
