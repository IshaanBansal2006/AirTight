from __future__ import annotations

from airtight.dimos_lane.modules.allocator import Allocator, AllocatorModule, verify_task
from airtight.dimos_lane.modules.dimos_backend import DimosBackend, DimosBackendModule
from airtight.dimos_lane.modules.fleet_memory import FleetMemoryModule
from airtight.dimos_lane.modules.gate import Gate, GateModule
from airtight.dimos_lane.modules.orchestrator import NORTH_GATE, Orchestrator
from airtight.dimos_lane.modules.sim_fleet import (
    SimFleet,
    SimFleetModule,
    default_drones,
    topic_for,
)
from airtight.dimos_lane.site_io import load_example_site


def test_each_module_blueprint_builds() -> None:
    from airtight.dimos_lane.modules.orchestrator import OrchestratorModule

    for cls in (
        AllocatorModule,
        GateModule,
        SimFleetModule,
        DimosBackendModule,
        FleetMemoryModule,
        OrchestratorModule,
    ):
        bp = cls.blueprint()
        assert bp is not None


def test_allocator_gives_verify_to_go2() -> None:
    alloc = Allocator()
    task = verify_task("verify-1", 60.0, 75.0)
    assignment = alloc.allocate([task], default_drones())
    winner = alloc.winner_for("verify-1")
    assert winner in {"go2_1", "guard_1"}
    assert any(task.task_id in [t.task_id for t in path] for path in assignment.values())


def test_gate_silence_never_authorizes() -> None:
    gate = Gate()
    pid = gate.propose("verify", "check north", deadline_s=5.0, now=10.0)
    assert pid in gate.pending_ids(now=10.0)
    expired = gate.pending_ids(now=16.0)
    assert pid not in expired
    events = [e["event"] for e in gate.audit()]
    assert "proposed" in events and "auto_denied_timeout" in events


def test_sim_fleet_prefixes_topics() -> None:
    fleet = SimFleet()
    fleet.set_pose("drone_1", 10.0, 20.0, 8.0)
    assert topic_for("drone_1") == "/drone_1/pose"
    assert fleet.snapshot()["drone_1"] == [10.0, 20.0, 8.0]


def test_dimos_backend_goto_and_step() -> None:
    calls: list[tuple[float, float]] = []
    backend = DimosBackend(move_to=lambda x, y: calls.append((x, y)) or "ok")
    import numpy as np

    backend.goto("go2_1", np.array([60.0, 75.0]))
    assert calls == [(60.0, 75.0)]
    backend.goto("drone_1", np.array([1.0, 2.0, 8.0]))
    start = backend.pose("drone_1").copy()
    backend.step(0.5)
    moved = backend.pose("drone_1")
    assert float(moved[0] - start[0]) != 0.0 or float(moved[1] - start[1]) != 0.0


def test_orchestrator_check_north_gate() -> None:
    orch = Orchestrator()
    site = load_example_site()
    gate = site.entry(NORTH_GATE)
    result = orch.dispatch_verify(gate.position.x, gate.position.y)
    assert "winner=" in result
    assert orch.last_dispatch in {"go2_1", "guard_1"}
    assert "example-yard" in orch.fleet_status()
