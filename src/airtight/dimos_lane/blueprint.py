"""airtight-site blueprint: slim Go2 + WalkModule site brain + MCP + one cheap LLM."""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect

from airtight.dimos_lane.modules.allocator import AllocatorModule
from airtight.dimos_lane.modules.dimos_backend import DimosBackendModule
from airtight.dimos_lane.modules.fleet_memory import FleetMemoryModule
from airtight.dimos_lane.modules.gate import GateModule
from airtight.dimos_lane.modules.orchestrator import OrchestratorModule
from airtight.dimos_lane.modules.sim_fleet import SimFleetModule

A6_MODEL = "gpt-4o-mini"
SITE_AGENT_PROMPT = """You operate a Unitree Go2 on the Airtight example yard.
Dock A is (10, 70). The north gate is (60, 75).

Use tools. If the user asks to check the north gate, call check_north_gate.
Otherwise call dispatch_verify(x, y) for a ground verify. Call fleet_status
to report poses. Do not call walk_forward or snapshot_camera unless asked.
After a dispatch, reply with the winner and task id in one sentence.
"""


def modules_only() -> Any:
    """Site stack without a Go2 child: wrappers, memory, orchestrator."""
    return autoconnect(
        AllocatorModule.blueprint(),
        GateModule.blueprint(),
        SimFleetModule.blueprint(),
        DimosBackendModule.blueprint(),
        FleetMemoryModule.blueprint(),
        OrchestratorModule.blueprint(),
    )


def build_airtight_site_modules() -> Any:
    """MCP without the Go2/SpeakSkill stack, so skills list without OpenAI."""
    from dimos.agents.mcp.mcp_server import McpServer

    return autoconnect(modules_only(), McpServer.blueprint())


def build_go2_sim() -> Any:
    """Go2 in MuJoCo plus MCP walk_forward. Fits in 8 GiB; no SpeakSkill/LLM."""
    from dimos.agents.mcp.mcp_server import McpServer
    from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic

    from airtight.dimos_lane.modules.walk import WalkModule

    return autoconnect(
        unitree_go2_basic,
        WalkModule.blueprint(),
        McpServer.blueprint(),
    )


def build_airtight_site() -> Any:
    """A6: slim Go2 + site skills + MCP server/client. Not unitree-go2-agentic."""
    from dimos.agents.mcp.mcp_client import McpClient
    from dimos.agents.mcp.mcp_server import McpServer
    from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic

    from airtight.dimos_lane.modules.walk import WalkModule
    from airtight.dimos_lane.openai_env import ensure_openai_key

    ensure_openai_key()
    return autoconnect(
        unitree_go2_basic,
        WalkModule.blueprint(),
        McpServer.blueprint(),
        McpClient.blueprint(model=A6_MODEL, system_prompt=SITE_AGENT_PROMPT),
    )


# Discoverable as `airtight.airtight-site-modules` — no OpenAI, no MuJoCo.
airtight_site_modules = build_airtight_site_modules()
airtight_go2 = build_go2_sim()


def __getattr__(name: str) -> Any:
    if name == "airtight_site":
        return build_airtight_site()
    raise AttributeError(name)
