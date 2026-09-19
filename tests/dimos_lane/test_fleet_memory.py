from __future__ import annotations

from airtight.dimos_lane.modules.fleet_memory import LocalFleetMemory


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


def test_delta_respects_byte_budget() -> None:
    mem = LocalFleetMemory()
    for i in range(40):
        mem.observe({"kind": "coverage", "x": float(i * 2), "y": 0.0, "last_seen": float(i)})
    small = mem.delta(0, 200)
    assert len(small) <= 200
