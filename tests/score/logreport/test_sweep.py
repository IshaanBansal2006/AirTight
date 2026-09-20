from __future__ import annotations

import json
from pathlib import Path

import pytest

from airtight.contracts import FleetConfig, Report, SensorCurves, Site, Tactic
from airtight.score.logreport.logs import summarize_log
from airtight.score.logreport.sweep import (
    ConfigInputs,
    build_report,
    coverage_gap,
    inputs_without_quiet,
    run_config,
    run_quiet_nights,
    seed_list_hash,
)
from airtight.sim.runner import run_episode

SCEN = Path(__file__).parents[3] / "scenarios" / "logistics_yard"


@pytest.fixture(autouse=True)
def _v0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRTIGHT_ENGINE", "v0")


def _scene() -> tuple[Site, SensorCurves]:
    return Site.model_validate_json(
        (SCEN / "site.json").read_text()
    ), SensorCurves.model_validate_json((SCEN / "sensor_curve.json").read_text())


def test_v0_log_summary_has_score_series(tmp_path: Path) -> None:
    site, curves = _scene()
    fleet = FleetConfig.model_validate_json(
        (SCEN / "fleets" / "d2_go2_guard_sync.json").read_text()
    )
    tactic = Tactic(
        id="walk",
        family="charging_window",
        entry_id="main_gate",
        phase=0.1,
        speed_mps=1.2,
        waypoints=[site.asset],
    )
    res = run_episode(site, fleet, tactic, curves, 3, tmp_path)
    s = summarize_log(res.log_path)
    assert (
        s.has_score_series
        and s.tactic_id == "walk"
        and s.t_end_s > 0
        and s.timely_at_ref == res.timely_detected
    )


def test_quiet_nights_and_coverage_gap_from_engine() -> None:
    site, curves = _scene()
    fleet = FleetConfig.model_validate_json(
        (SCEN / "fleets" / "d2_go2_guard_sync.json").read_text()
    )
    quiet = run_quiet_nights(site, fleet, curves, [11], workers=1)
    assert quiet.hours > 0.5 and "quiet nights" in quiet.source
    gap = coverage_gap(site, fleet, curves)
    assert 0.0 <= gap <= 3600.0


def test_two_config_sweep_builds_a_valid_report(tmp_path: Path) -> None:
    site, curves = _scene()
    tactics = [
        Tactic(
            id="walk",
            family="charging_window",
            entry_id="main_gate",
            phase=0.1,
            speed_mps=1.2,
            waypoints=[site.asset],
        )
    ]
    seeds = [1, 2, 3]
    per_config: dict[str, ConfigInputs] = {}
    for name in ("d2_go2_guard_sync", "d4_go2_guard_stagger"):
        fleet = FleetConfig.model_validate_json((SCEN / "fleets" / f"{name}.json").read_text())
        summaries = run_config(
            site, fleet, tactics, curves, seeds, run_episode, tmp_path / "logs", prune_logs=True
        )
        blind = run_config(
            site,
            fleet,
            tactics,
            curves,
            seeds,
            run_episode,
            tmp_path / "logs",
            prune_logs=True,
            randomize_phase=True,
        )
        assert [b.tactic_id for b in blind] == ["walk"] * 3
        per_config[name] = inputs_without_quiet(fleet, summaries).model_copy(
            update={"summaries_blind": blind}
        )
    assert not (tmp_path / "logs" / "d2_go2_guard_sync").exists()
    assert not (tmp_path / "logs" / "d2_go2_guard_sync_blind").exists()
    report = build_report(
        site,
        per_config,
        "d2_go2_guard_sync",
        1.0,
        seeds,
        seed_list_hash(seeds),
        {"adversary_knowledge": "a", "sensor_calibration": "b", "detection_model_note": "c"},
    )
    assert Report.model_validate_json(report.model_dump_json()) == report
    other = report.config("d4_go2_guard_stagger")
    assert other.worst_tactic_pd_schedule_blind is not None
    assert 0.0 <= other.worst_tactic_pd_schedule_blind <= 1.0
    assert other.n_episodes == 3 and {d.metric for d in other.paired_vs_baseline} == {
        "pd_at_operating_point",
        "human_decisions_per_hour",
        "coverage_gap_s_per_hour",
    }
    assert json.loads(report.model_dump_json())["conditions"]["n_seeds"] == 3


def test_v0_pruned_sweep_uses_live_peaks_and_writes_nothing(tmp_path: Path) -> None:
    site, curves = _scene()
    fleet = FleetConfig.model_validate_json(
        (SCEN / "fleets" / "d2_go2_guard_sync.json").read_text()
    )
    tactic = Tactic(
        id="walk",
        family="charging_window",
        entry_id="main_gate",
        phase=0.1,
        speed_mps=1.2,
        waypoints=[site.asset],
    )
    summaries = run_config(
        site, fleet, [tactic], curves, [3], run_episode, tmp_path / "logs", prune_logs=True
    )
    assert summaries[0].has_score_series
    assert summaries[0].intruder_peak_before_cdp not in (float("inf"), float("-inf"))
    assert not (tmp_path / "logs").exists() or not any((tmp_path / "logs").rglob("*.jsonl"))


def test_summarize_from_scores_matches_full_log(tmp_path: Path) -> None:
    from airtight.score.logreport.logs import summarize_from_scores
    from airtight.sim.episode import official_params, simulate

    site, curves = _scene()
    fleet = FleetConfig.model_validate_json(
        (SCEN / "fleets" / "d2_go2_guard_sync.json").read_text()
    )
    tactic = Tactic(
        id="walk",
        family="charging_window",
        entry_id="main_gate",
        phase=0.1,
        speed_mps=1.2,
        waypoints=[site.asset],
    )
    scores = simulate(site, fleet, tactic, curves, 3, official_params())
    from_scores = summarize_from_scores(fleet, tactic, scores)
    from_log = summarize_log(run_episode(site, fleet, tactic, curves, 3, tmp_path).log_path)
    assert from_scores.intruder_peak_before_cdp == from_log.intruder_peak_before_cdp
    assert from_scores.timely_at_ref == from_log.timely_at_ref
    assert from_scores.has_score_series == from_log.has_score_series

