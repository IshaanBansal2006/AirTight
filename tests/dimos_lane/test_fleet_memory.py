from __future__ import annotations

from typing import TYPE_CHECKING

from airtight.dimos_lane.modules.fleet_memory import (
    FleetMemoryModule,
    LocalFleetMemory,
    format_recall,
    load_fleet_memory,
    remember_item,
)

if TYPE_CHECKING:
    import pytest


def test_merge_is_idempotent_and_order_independent() -> None:
    a = LocalFleetMemory()
    b = LocalFleetMemory()
    a.observe({"kind": "coverage", "x": 4.0, "y": 6.0, "last_seen": 10.0})
    a.observe({"kind": "evidence", "key": "obs-1", "score": 1.5})
    b.observe({"kind": "coverage", "x": 4.0, "y": 6.0, "last_seen": 12.0})
    b.observe({"kind": "evidence", "key": "obs-1", "score": 1.5})
    b.observe({"kind": "claim", "key": "north", "t": 3.0, "value": "go2_1"})

    ab = LocalFleetMemory()
    ab.merge(a.delta(0, 10_000))
    ab.merge(b.delta(0, 10_000))
    ba = LocalFleetMemory()
    ba.merge(b.delta(0, 10_000))
    ba.merge(a.delta(0, 10_000))
    ab.merge(b.delta(0, 10_000))  # idempotent second merge

    cov_a = {(item["x"], item["y"], item["last_seen"]) for item in ab.query("coverage")}
    cov_b = {(item["x"], item["y"], item["last_seen"]) for item in ba.query("coverage")}
    assert cov_a == cov_b
    assert ab.evidence_score() == ba.evidence_score() == 1.5
    claims = ab.query("claim")
    assert claims[0]["value"] == "go2_1"


def test_load_prefers_lane_c_store(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyStore:
        def observe(self, item: object) -> None:
            return None

        def delta(self, since_version: int, byte_budget: int) -> bytes:
            return b""

        def merge(self, delta: bytes) -> None:
            return None

        def query(self, kind: str, region: object = None) -> list[object]:
            return []

        @property
        def version(self) -> int:
            return 0

    import airtight.memory as mem

    monkeypatch.setattr(mem, "FleetMemoryStore", DummyStore, raising=False)
    inner = load_fleet_memory()
    assert isinstance(inner, DummyStore)


def test_delta_respects_byte_budget() -> None:
    mem = LocalFleetMemory()
    for i in range(40):
        mem.observe({"kind": "coverage", "x": float(i * 2), "y": 0.0, "last_seen": float(i)})
    small = mem.delta(0, 200)
    assert len(small) <= 200


def test_recall_returns_entries_with_age() -> None:
    mem = LocalFleetMemory()
    remember_item(mem, kind="coverage", x=10.0, y=70.0, t=0.0)
    remember_item(mem, kind="claim", key="last_dispatch", value="go2_1", t=0.0)
    coverage = format_recall(mem, "*", "coverage")
    claims = format_recall(mem, "*", "claim")
    assert "age=" in coverage
    assert "age=" in claims
    assert "go2_1" in claims
    aged = mem.query("coverage")
    assert aged and aged[0]["age"] > 0
    empty = format_recall(mem, "0,0,1,1", "coverage")
    assert "empty" in empty


def test_recall_and_remember_are_skills() -> None:
    assert getattr(FleetMemoryModule.recall, "__skill__", False)
    assert getattr(FleetMemoryModule.remember, "__skill__", False)


def test_load_uses_c_store() -> None:
    from airtight.memory.store import FleetMemoryStore

    inner = load_fleet_memory()
    assert isinstance(inner, FleetMemoryStore)


def test_recall_on_c_store_includes_age() -> None:
    from airtight.memory.store import FleetMemoryStore

    store = FleetMemoryStore(cell_m=2.0)
    remember_item(store, kind="coverage", x=4.0, y=6.0, t=10.0, key="drone_1")
    remember_item(store, kind="claim", key="north", value="go2_1", t=3.0)
    coverage = format_recall(store, "*", "coverage")
    claims = format_recall(store, "*", "claim")
    assert "age=" in coverage
    assert "go2_1" in claims and "age=" in claims
    recs = store.records("claim", now=5.0)
    assert recs[0]["agent_id"] == "go2_1" and recs[0]["age"] == 2.0
    assert store.records("coverage")[0]["by"] == "drone_1"
