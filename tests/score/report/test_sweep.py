from __future__ import annotations

import json
from pathlib import Path

import pytest

from airtight.contracts import FleetConfig, Report, SensorCurves, Site, Tactic
from airtight.score.report.logs import summarize_log
from airtight.score.report.sweep import build_report, run_config, seed_list_hash
from airtight.sim.runner import run_episode

SCEN = Path(__file__).parents[3] / "scenarios" / "logistics_yard"


@pytest.fixture(autouse=True)
def _v0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRTIGHT_ENGINE", "v0")


def test_v0_log_summary_has_score_series(tmp_path: Path) -> None:
    site = Site.model_validate_json((SCEN / "site.json").read_text())
    curves = SensorCurves.model_validate_json((SCEN / "sensor_curve.json").read_text())
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


def test_two_config_sweep_builds_a_valid_report(tmp_path: Path) -> None:
    site = Site.model_validate_json((SCEN / "site.json").read_text())
    curves = SensorCurves.model_validate_json((SCEN / "sensor_curve.json").read_text())
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
    per_config = {}
    for name in ("d2_go2_guard_sync", "d4_go2_guard_stagger"):
        fleet = FleetConfig.model_validate_json((SCEN / "fleets" / f"{name}.json").read_text())
        per_config[name] = (
            fleet,
            run_config(
                site, fleet, tactics, curves, seeds, run_episode, tmp_path / "logs", prune_logs=True
            ),
        )
    assert not (tmp_path / "logs" / "d2_go2_guard_sync").exists()
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
    assert (
        other.n_episodes == 3
        and other.paired_vs_baseline
        and other.paired_vs_baseline[0].metric == "pd_at_operating_point"
    )
    assert json.loads(report.model_dump_json())["conditions"]["n_seeds"] == 3
