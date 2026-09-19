"""airtight-site blueprint: Go2 agentic stack plus lane A modules."""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect

from airtight.dimos_lane.modules.allocator import AllocatorModule
from airtight.dimos_lane.modules.dimos_backend import DimosBackendModule
from airtight.dimos_lane.modules.gate import GateModule
from airtight.dimos_lane.modules.orchestrator import OrchestratorModule
from airtight.dimos_lane.modules.sim_fleet import SimFleetModule


def modules_only() -> Any:
    """Four wrappers plus orchestrator, no Go2 child process."""
    return autoconnect(
        AllocatorModule.blueprint(),
        GateModule.blueprint(),
        SimFleetModule.blueprint(),
        DimosBackendModule.blueprint(),
        OrchestratorModule.blueprint(),
    )


def build_airtight_site(*, with_go2: bool = True) -> Any:
    stack = modules_only()
    if not with_go2:
        return stack
    from dimos.robot.unitree.go2.blueprints.agentic.unitree_go2_agentic import (
        unitree_go2_agentic,
    )

    return autoconnect(unitree_go2_agentic, stack)


def __getattr__(name: str) -> Any:
    if name == "airtight_site":
        return build_airtight_site(with_go2=True)
    raise AttributeError(name)
