from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

PITCH = Path(__file__).parents[1] / "pitch"


@pytest.fixture(autouse=True)
def _pitch_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(PITCH))


def test_make_charts_renders_from_example_report(tmp_path: Path) -> None:
    sys.argv = [
        "make_charts.py",
        "--report",
        str(tmp_path / "missing.json"),
        "--tactics-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "charts"),
    ]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "make_charts.py"), run_name="__main__")
    assert exc.value.code == 0
    names = {p.name for p in (tmp_path / "charts").iterdir()}
    assert {
        "cost_vs_detection.png",
        "roc.png",
        "vulnerability_map.png",
        "before_after.png",
        "numbers.json",
    } <= names
    numbers = json.loads((tmp_path / "charts" / "numbers.json").read_text())
    assert numbers["before_after"]["fixed"] == "3drone_go2_stagger" and numbers["frontier"]


def test_make_token_chart_without_ledger(tmp_path: Path) -> None:
    sys.argv = [
        "make_token_chart.py",
        "--ledger",
        str(tmp_path / "none.jsonl"),
        "--out",
        str(tmp_path / "charts"),
    ]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "make_token_chart.py"), run_name="__main__")
    assert exc.value.code == 0
    numbers = json.loads((tmp_path / "charts" / "token_numbers.json").read_text())
    assert (
        numbers["comparison"]["ratio"] == pytest.approx(100.0)
        and "price table" in numbers["comparison"]["basis"]
    )


def test_build_deck_from_rendered_numbers(tmp_path: Path) -> None:
    charts = tmp_path / "charts"
    sys.argv = [
        "make_charts.py",
        "--report",
        str(tmp_path / "missing.json"),
        "--tactics-dir",
        str(tmp_path),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_charts.py"), run_name="__main__")
    sys.argv = [
        "make_token_chart.py",
        "--ledger",
        str(tmp_path / "none.jsonl"),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_token_chart.py"), run_name="__main__")
    sys.argv = ["build_deck.py", "--charts", str(charts), "--out", str(tmp_path / "deck.md")]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "build_deck.py"), run_name="__main__")
    assert exc.value.code == 0
    deck = (tmp_path / "deck.md").read_text()
    assert deck.count("\n---\n") >= 8 and "EXAMPLE DATA" in deck and "cost_vs_detection.png" in deck


def test_render_replay_from_a_v0_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
    from airtight.sim.runner import run_episode

    monkeypatch.setenv("AIRTIGHT_ENGINE", "v0")
    scen = Path(__file__).parents[1] / "scenarios" / "logistics_yard"
    site = Site.model_validate_json((scen / "site.json").read_text())
    curves = SensorCurves.model_validate_json((scen / "sensor_curve.json").read_text())
    fleet = FleetConfig.model_validate_json(
        (scen / "fleets" / "d2_go2_guard_sync.json").read_text()
    )
    tactic = Tactic(
        id="walk",
        family="charging_window",
        entry_id="main_gate",
        phase=0.1,
        speed_mps=1.5,
        waypoints=[site.asset],
    )
    res = run_episode(site, fleet, tactic, curves, 5, tmp_path / "logs")
    sys.argv = [
        "render_replay.py",
        str(res.log_path),
        "--site",
        str(scen / "site.json"),
        "--out",
        str(tmp_path / "clip.mp4"),
        "--fps",
        "4",
        "--speed",
        "20",
    ]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "render_replay.py"), run_name="__main__")
    assert exc.value.code == 0 and (tmp_path / "clip.mp4").stat().st_size > 1000


def test_fill_writeup_from_rendered_numbers(tmp_path: Path) -> None:
    charts = tmp_path / "charts"
    sys.argv = [
        "make_charts.py",
        "--report",
        str(tmp_path / "missing.json"),
        "--tactics-dir",
        str(tmp_path),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_charts.py"), run_name="__main__")
    sys.argv = [
        "make_token_chart.py",
        "--ledger",
        str(tmp_path / "none.jsonl"),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_token_chart.py"), run_name="__main__")
    sys.argv = ["fill_writeup.py", "--charts", str(charts), "--out", str(tmp_path / "w.md")]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "fill_writeup.py"), run_name="__main__")
    assert exc.value.code == 0
    text = (tmp_path / "w.md").read_text()
    assert "EXAMPLE DATA" in text and "[pd_baseline]" not in text and "[ratio]" not in text
