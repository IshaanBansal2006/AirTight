"""Live A1–A3 checks against a running `airtight.airtight-go2` daemon.

Skipped unless MCP is already up or AIRTIGHT_LIVE=1.
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003

import pytest
import requests

from airtight.dimos_lane.person import default_intruder_xy, place_intruder, publish_person_pose

LIVE = pytest.mark.live


def _adapter():
    from dimos.agents.mcp.mcp_adapter import McpAdapter

    return McpAdapter.from_run_entry(timeout=30)


def _mcp_up() -> bool:
    try:
        adapter = _adapter()
        adapter.list_tools()
        return True
    except (requests.ConnectionError, OSError, Exception):
        return False


pytestmark = pytest.mark.skipif(
    os.environ.get("AIRTIGHT_LIVE") != "1" and not _mcp_up(),
    reason="no airtight.airtight-go2 MCP server",
)


def _tool_names(tools: list[dict[str, object]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _call_text(name: str, arguments: dict[str, object] | None = None) -> str:
    adapter = _adapter()
    return adapter.call_tool_text(name, arguments)


@LIVE
def test_a1_lists_walk_and_battery() -> None:
    tools = _tool_names(_adapter().list_tools())
    assert "walk_forward" in tools
    assert "get_battery_soc" in tools
    assert "snapshot_camera" in tools


@LIVE
def test_a1_walk_forward_reports_motion() -> None:
    text = _call_text("walk_forward", {"seconds": 2.0, "vx": 0.40})
    assert "2.0s" in text
    assert "0.40 m/s" in text
    assert "displacement=" in text
    value = float(text.split("displacement=")[1].split("m")[0])
    assert value >= 0.02


@LIVE
def test_a1_get_battery_soc_does_not_crash() -> None:
    text = _call_text("get_battery_soc")
    # Sim has no BMS: None/null is the documented answer.
    assert text.strip() in {"None", "null", ""} or "None" in text or "null" in text.lower()


@LIVE
def test_a3_camera_changes_when_intruder_is_placed(tmp_path: Path) -> None:
    import time

    empty = tmp_path / "empty.jpg"
    filled = tmp_path / "intruder.jpg"
    artifact = Path(__file__).resolve().parents[2] / "data" / "go2_intruder.jpg"
    publish_person_pose(0.0, 0.0, z=0.0)
    time.sleep(1.0)
    _call_text("snapshot_camera", {"path": str(empty)})
    xy = place_intruder()
    expected = default_intruder_xy()
    assert abs(xy.x - expected.x) < 1e-6
    time.sleep(1.5)
    _call_text("snapshot_camera", {"path": str(filled)})
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(filled.read_bytes())
    assert empty.is_file() and filled.is_file()
    assert empty.stat().st_size > 1000
    assert filled.stat().st_size > 1000
    assert empty.read_bytes() != filled.read_bytes()
