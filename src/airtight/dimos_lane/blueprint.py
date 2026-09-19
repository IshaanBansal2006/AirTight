"""airtight-site blueprint: Go2 agentic stack plus lane A modules."""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect

from airtight.dimos_lane.modules.allocator import AllocatorModule
from airtight.dimos_lane.modules.dimos_backend import DimosBackendModule
from airtight.dimos_lane.modules.fleet_memory import FleetMemoryModule
from airtight.dimos_lane.modules.gate import GateModule
from airtight.dimos_lane.modules.orchestrator import OrchestratorModule
from airtight.dimos_lane.modules.sim_fleet import SimFleetModule


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


def build_airtight_site(*, with_go2: bool = True) -> Any:
    stack = modules_only()
    if not with_go2:
        from dimos.agents.mcp.mcp_server import McpServer

        # MCP without the Go2/SpeakSkill stack, so A0 can list skills without OpenAI.
        return autoconnect(stack, McpServer.blueprint())
    from dimos.robot.unitree.go2.blueprints.agentic.unitree_go2_agentic import (
        unitree_go2_agentic,
    )

    return autoconnect(unitree_go2_agentic, stack)


# Discoverable as `airtight.airtight-site-modules` — no OpenAI, no MuJoCo.
airtight_site_modules = build_airtight_site(with_go2=False)


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


airtight_go2 = build_go2_sim()


def __getattr__(name: str) -> Any:
    if name == "airtight_site":
        return build_airtight_site(with_go2=True)
    raise AttributeError(name)
