"""Site orchestrator skills: dispatch_verify and fleet_status."""

from __future__ import annotations

import numpy as np
from dimos.agents.annotation import skill
from dimos.core.module import Module

from airtight.dimos_lane.modules.allocator import Allocator, verify_task
from airtight.dimos_lane.modules.dimos_backend import DimosBackend
from airtight.dimos_lane.modules.gate import Gate
from airtight.dimos_lane.modules.sim_fleet import SimFleet
from airtight.dimos_lane.site_io import load_example_site

NORTH_GATE = "north_gate"


class Orchestrator:
    def __init__(self) -> None:
        self.allocator = Allocator()
        self.gate = Gate()
        self.fleet = SimFleet()
        self.backend = DimosBackend()
        self.last_dispatch: str | None = None

    def dispatch_verify(self, x: float, y: float) -> str:
        task = verify_task("verify-north" if (x, y) == _north_gate_xy() else f"verify-{x:.0f}-{y:.0f}", x, y)
        assignment = self.allocator.allocate([task], self.fleet.drones())
        winner = self.allocator.winner_for(task.task_id)
        if winner is None:
            return "no capable agent for verify"
        self.backend.goto(winner, np.array([x, y], dtype=np.float64))
        proposal = self.gate.propose(
            action="verify",
            rationale=f"dispatch {winner} to ({x:.1f},{y:.1f})",
        )
        self.last_dispatch = winner
        assigned = {did: [t.task_id for t in path] for did, path in assignment.items() if path}
        return f"task={task.task_id} winner={winner} auction={assigned} proposal={proposal}"

    def fleet_status(self) -> str:
        site = load_example_site()
        poses = self.fleet.snapshot()
        agents = ", ".join(f"{did}=({p[0]:.0f},{p[1]:.0f})" for did, p in poses.items())
        pending = self.gate.pending_ids()
        return (
            f"site {site.name}; agents [{agents}]; "
            f"last_dispatch={self.last_dispatch}; pending={pending or 'none'}"
        )


def _north_gate_xy() -> tuple[float, float]:
    site = load_example_site()
    gate = site.entry(NORTH_GATE)
    return gate.position.x, gate.position.y


class OrchestratorModule(Module):
    def __init__(self, config_args: dict[str, object] | None = None) -> None:
        super().__init__(dict(config_args or {}))
        self.inner = Orchestrator()

    @skill
    def dispatch_verify(self, x: float, y: float) -> str:
        """Create a verify task at (x, y), auction it, send the winner."""
        return self.inner.dispatch_verify(x, y)

    @skill
    def fleet_status(self) -> str:
        """Summarize agent poses, last dispatch and pending approvals."""
        return self.inner.fleet_status()

    @skill
    def site_status(self) -> str:
        """Describe the loaded site: name, entries and docks."""
        site = load_example_site()
        entries = ", ".join(e.id for e in site.entry_points)
        return (
            f"site {site.name}: {len(site.entry_points)} entry points ({entries}), "
            f"{len(site.docks)} docks, response time {site.response_time_s:.0f}s"
        )

    @skill
    def check_north_gate(self) -> str:
        """Convenience for the demo prompt 'check the north gate'."""
        x, y = _north_gate_xy()
        return self.inner.dispatch_verify(x, y)
