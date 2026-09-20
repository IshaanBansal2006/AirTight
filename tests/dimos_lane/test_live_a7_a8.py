"""Live A7–A8 checks against a running airtight-site daemon.

Skipped unless MCP is already up or AIRTIGHT_LIVE=1. A8 live dispatch only
runs when AIRTIGHT_LIVE=1 so a leftover daemon is not walked by accident.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import pytest
import requests

LIVE = pytest.mark.live


def _adapter():
    from dimos.agents.mcp.mcp_adapter import McpAdapter

    return McpAdapter.from_run_entry(timeout=60)


def _mcp_up() -> bool:
    try:
        _adapter().list_tools()
        return True
    except (requests.ConnectionError, OSError, Exception):
        return False


pytestmark = pytest.mark.skipif(
    os.environ.get("AIRTIGHT_LIVE") != "1" and not _mcp_up(),
    reason="no airtight MCP server",
)


def _tool_names(tools: list[dict[str, object]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _call_text(name: str, arguments: dict[str, object] | None = None) -> str:
    return _adapter().call_tool_text(name, arguments)


@LIVE
def test_a7_recall_returns_age() -> None:
    tools = _tool_names(_adapter().list_tools())
    if "recall" not in tools:
        pytest.skip("daemon predates WalkModule.recall; restart airtight.airtight-site")
    _call_text("fleet_status")
    text = _call_text("recall", {"query": "*", "kind": "coverage"})
    assert "age=" in text
    claims = _call_text("recall", {"query": "*", "kind": "claim"})
    assert "kind=claim" in claims or "age=" in claims or "empty" in claims


@LIVE
def test_a8_replay_dispatches_via_mcp() -> None:
    if os.environ.get("AIRTIGHT_LIVE") != "1":
        pytest.skip("set AIRTIGHT_LIVE=1 to script /person_pose and dispatch")
    tools = _tool_names(_adapter().list_tools())
    if "dispatch_verify" not in tools:
        pytest.skip("daemon is airtight-go2; restart as airtight.airtight-site")
    from airtight.dimos_lane.replay import dispatch_via_mcp, plan_replay, run_replay

    example = Path(str(resources.files("airtight.contracts.examples").joinpath("episode.jsonl")))
    plan = plan_replay(example, dispatch=False)
    result = run_replay(
        plan,
        live=True,
        realtime_scale=0.0,
        dispatch=dispatch_via_mcp,
        densify=False,
    )
    assert result["dispatch"] == 1
    assert result["dispatch_result"] is not None
    assert "winner=" in result["dispatch_result"]
