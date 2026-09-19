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
