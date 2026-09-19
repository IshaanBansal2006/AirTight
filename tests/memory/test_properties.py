from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from airtight.memory import Claim, CoverageCell, Evidence, FleetMemoryStore

_t = st.floats(min_value=0, max_value=1e4, allow_nan=False, allow_infinity=False)
_agent = st.sampled_from(["drone_1", "drone_2", "go2_1", "guard_1"])

cells = st.builds(
    CoverageCell, cx=st.integers(0, 5), cy=st.integers(0, 5), last_seen_t=_t, by=_agent
)
claims = st.builds(
    Claim, task_id=st.sampled_from(["verify_1", "verify_2", "patrol_3"]), agent_id=_agent, t=_t
)
evidence = st.builds(
    Evidence,
    evidence_id=st.sampled_from([f"e{i}" for i in range(8)]),
    object_id=st.sampled_from(["intruder", "delivery", "staff"]),
    agent_id=_agent,
    t=_t,
    score=st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False),
    x=st.floats(0, 120, allow_nan=False),
    y=st.floats(0, 80, allow_nan=False),
)
items = st.one_of(cells, claims, evidence)


def _filled(seq: list) -> FleetMemoryStore:  # type: ignore[type-arg]
    s = FleetMemoryStore()
    for it in seq:
        s.observe(it)
    return s


@settings(max_examples=200)
@given(st.lists(items, max_size=30), st.randoms())
def test_merge_is_order_independent(seq: list, rnd) -> None:  # type: ignore[no-untyped-def, type-arg]
    shuffled = list(seq)
    rnd.shuffle(shuffled)
    assert _filled(seq).snapshot() == _filled(shuffled).snapshot()


@settings(max_examples=200)
@given(st.lists(items, max_size=30))
def test_merge_is_idempotent(seq: list) -> None:  # type: ignore[type-arg]
    once = _filled(seq)
    twice = _filled(seq + seq)
    assert once.snapshot() == twice.snapshot()


@settings(max_examples=200)
@given(st.lists(items, max_size=30), st.lists(items, max_size=30))
def test_delta_merge_converges_and_is_symmetric(a_items: list, b_items: list) -> None:  # type: ignore[type-arg]
    a, b = _filled(a_items), _filled(b_items)
    a.merge(b.delta(0, 1 << 20))
    b.merge(a.delta(0, 1 << 20))
    assert a.snapshot() == b.snapshot()
    v = a.version
    a.merge(b.delta(0, 1 << 20))
    assert a.version == v


@settings(max_examples=100)
@given(st.lists(items, min_size=1, max_size=30), st.integers(0, 600))
def test_delta_respects_budget_newest_first(seq: list, budget: int) -> None:  # type: ignore[type-arg]
    a = _filled(seq)
    d = a.delta(0, budget)
    assert len(d) <= budget
    sent = d.decode().splitlines()
    full = a.delta(0, 1 << 20).decode().splitlines()
    assert sent == full[: len(sent)]


def test_since_version_only_ships_changes() -> None:
    a = FleetMemoryStore()
    a.observe(Claim(task_id="verify_1", agent_id="go2_1", t=10))
    v = a.version
    a.observe(Claim(task_id="verify_1", agent_id="drone_1", t=5))
    assert a.delta(v, 1 << 20) == b""
    a.observe(CoverageCell(cx=1, cy=1, last_seen_t=3, by="drone_2"))
    assert b'"coverage"' in a.delta(v, 1 << 20)


def test_evidence_sums_and_dedupes() -> None:
    a = FleetMemoryStore()
    e = Evidence(
        evidence_id="e1", object_id="intruder", agent_id="drone_1", t=1, score=2.0, x=10, y=10
    )
    a.observe(e)
    a.observe(e)
    a.observe(
        Evidence(
            evidence_id="e2", object_id="intruder", agent_id="go2_1", t=2, score=1.5, x=12, y=11
        )
    )
    assert a.evidence_score("intruder") == 3.5
    assert (
        a.query("evidence", region=(0, 0, 11, 11))
        and len(a.query("evidence", region=(50, 50, 60, 60))) == 0
    )


def test_dict_items_from_lane_a_shape_are_accepted() -> None:
    a = FleetMemoryStore(cell_m=2.0)
    a.observe({"kind": "coverage", "x": 4.0, "y": 6.0, "last_seen": 10.0})
    a.observe({"kind": "evidence", "key": "obs-1", "score": 1.5})
    a.observe({"kind": "claim", "key": "north", "t": 3.0, "value": "go2_1"})
    assert a.query("coverage")[0].cx == 2 and a.query("coverage")[0].cy == 3
    assert a.evidence_score("unknown") == 1.5
    recs = a.records("claim", now=5.0)
    assert recs[0]["agent_id"] == "go2_1" and recs[0]["age"] == 2.0
    assert a.records("coverage")[0]["x"] == 5.0
