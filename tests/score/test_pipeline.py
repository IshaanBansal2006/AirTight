"""The one-command pipeline: argument routing and stopping on a failed step."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import pytest

from airtight.score import pipeline

if TYPE_CHECKING:
    from pathlib import Path


def _fake(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, list[str]]], codes: dict[str, int]
) -> None:
    for step, name in pipeline.MODULES.items():
        module = types.ModuleType(name)

        def main(argv: list[str], step: str = step) -> int:
            calls.append((step, argv))
            return codes.get(step, 0)

        module.main = main  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)


def test_runs_every_step_in_order_and_routes_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str]]] = []
    _fake(monkeypatch, calls, {})
    out = tmp_path / "campaign"
    code = pipeline.main(
        ["--out", str(out), "--deadline-hours", "2", "--tactics-dir", str(tmp_path), "--smoke"]
    )
    assert code == 0 and [step for step, _ in calls] == list(pipeline.STEPS)
    campaign_argv = calls[0][1]
    assert campaign_argv[:4] == ["--out", str(out), "--deadline-hours", "2.0"]
    assert "--smoke" in campaign_argv and str(tmp_path) in campaign_argv
    assert calls[1][1] == ["--out", str(out / "smoke")] == calls[2][1]


def test_stops_at_the_first_failing_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    _fake(monkeypatch, calls, {"report": 3})
    assert pipeline.main(["--out", str(tmp_path / "c")]) == 3
    assert [step for step, _ in calls] == ["campaign", "report"]


def test_rejects_relative_out_and_unknown_steps(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        pipeline.main(["--out", "relative"])
    with pytest.raises(SystemExit):
        pipeline.main(["--out", str(tmp_path), "--steps", "campaign,magic"])
