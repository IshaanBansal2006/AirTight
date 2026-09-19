"""Live A4–A6 checks against a running airtight daemon.

A4 (calibration) works on airtight-go2 or airtight-site (needs snapshot_camera).
A5/A6 need WalkModule site skills; A6 also needs agent_send + McpClient.

Skipped unless MCP is already up or AIRTIGHT_LIVE=1.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import requests

from airtight.dimos_lane.calibration import RANGE_BINS_M, read_looks
from airtight.dimos_lane.calibration.fit import fit_cache
from airtight.dimos_lane.openai_env import ensure_openai_key, openai_auth_error

LIVE = pytest.mark.live
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _adapter():
    from dimos.agents.mcp.mcp_adapter import McpAdapter

    return McpAdapter.from_run_entry(timeout=60)


def _mcp_up() -> bool:
    try:
        adapter = _adapter()
        adapter.list_tools()
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
    adapter = _adapter()
    return adapter.call_tool_text(name, arguments)


def _has_site_skills() -> bool:
    tools = _tool_names(_adapter().list_tools())
    return {"dispatch_verify", "fleet_status", "check_north_gate"} <= tools


@LIVE
def test_a4_live_owlv2_sweep_fits_curve(tmp_path: Path) -> None:
    from airtight.dimos_lane.calibration.capture import run_live_from_mcp

    cache = DATA / "calibration_looks.jsonl"
    snapshot_dir = tmp_path / "frames"
    looks = run_live_from_mcp(cache, snapshot_dir, frames_per_cell=2, settle_s=0.4)
    person = [lk for lk in looks if lk.cls == "person"]
    vehicle = [lk for lk in looks if lk.cls == "vehicle"]
    assert {lk.range_m for lk in person} == set(RANGE_BINS_M)
    assert {lk.bearing_deg for lk in person} == {-30.0, 0.0, 30.0}
    assert len(person) == 8 * 3 * 2
    assert len(vehicle) == 4 * 2
    assert all(lk.detector == "owlv2" for lk in looks)
    assert len(read_looks(cache)) == len(looks)
    out = DATA / "sensor_curve.json"
    curves = fit_cache(cache, out)
    go2 = curves.curves["go2_camera"]
    assert go2.range_bins_m == RANGE_BINS_M
    assert len(go2.pd_per_look) == 8
    assert "owlv2" in curves.source
    assert out.is_file()


@LIVE
def test_a5_check_north_gate_moves_go2() -> None:
    if not _has_site_skills():
        pytest.skip("daemon is airtight-go2; restart as airtight.airtight-site")
    before = _call_text("fleet_status")
    result = _call_text("check_north_gate")
    assert "verify-north" in result
    assert "winner=go2_1" in result
    after = _call_text("fleet_status")
    assert "last_dispatch=go2_1" in after
    assert "dispatches=" in after
    before_n = int(before.split("dispatches=")[1].split(";")[0]) if "dispatches=" in before else 0
    after_n = int(after.split("dispatches=")[1].split(";")[0])
    assert after_n == before_n + 1


@LIVE
def test_a6_agent_send_check_north_gate() -> None:
    if not _has_site_skills():
        pytest.skip("daemon is airtight-go2; restart as airtight.airtight-site")
    tools = _tool_names(_adapter().list_tools())
    assert "agent_send" in tools
    assert ensure_openai_key(), "OPENAI_API_KEY missing; A6 needs one gpt-4o-mini turn"
    auth = openai_auth_error()
    if auth is not None:
        pytest.fail(f"OpenAI rejected the key ({auth}); A6 needs a valid gpt-4o-mini key")
    before = _call_text("fleet_status")
    before_n = int(before.split("dispatches=")[1].split(";")[0]) if "dispatches=" in before else 0
    if "prompt_agent" in tools:
        sent = _call_text("prompt_agent", {"message": "check the north gate"})
        assert "queued" in sent.lower() or "check the north gate" in sent.lower()
    else:
        sent = _call_text("agent_send", {"message": "check the north gate"})
        assert "check the north gate" in sent.lower()
    deadline = time.time() + 90.0
    last = before
    while time.time() < deadline:
        last = _call_text("fleet_status")
        if "dispatches=" in last:
            n = int(last.split("dispatches=")[1].split(";")[0])
            if n > before_n and "last_dispatch=go2_1" in last:
                return
        time.sleep(2.0)
    pytest.fail(f"A6 LLM did not dispatch go2_1 within 90s; last status={last}")
