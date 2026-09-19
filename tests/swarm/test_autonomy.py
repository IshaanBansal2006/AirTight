"""Step-1 L2 tests: decomposer (HTN), behavior trees, CBBA allocation, and the
full intent -> decompose -> allocate pipeline."""

from __future__ import annotations

import pytest

from airtight.swarm.autonomy.allocator import CBBAAllocator
from airtight.swarm.autonomy.bt import Action, Fallback, Sequence, Status
from airtight.swarm.autonomy.decomposer import HTNDecomposer, LLMDecomposer, Task
from airtight.swarm.autonomy.world_state import DroneState, WorldState
from airtight.swarm.schemas import StructuredIntent, TrackMsg


def make_world(n_drones: int = 2) -> WorldState:
    w = WorldState()
    for i in range(n_drones):
        w.update_drone(
            DroneState(
                drone_id=f"d{i}",
                position=[float(5 * i), 0.0, 2.0],
                capabilities=frozenset({"camera", "radar"}),
            )
        )
    w.update_tracks(
        [
            TrackMsg(
                track_id=7,
                timestamp=1.0,
                position=[3, 4, 0.5],
                velocity=[0.5, 0.2, 0],
                extent=[0.2] * 3,
                position_sqrt_cov=[0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
            )
        ],
        timestamp=1.0,
    )
    return w


# ------------------------------------------------------------------ decomposer
def test_htn_goto_track_patrol() -> None:
    htn = HTNDecomposer()
    w = make_world()
    goto = htn.decompose(StructuredIntent(intent_id="i1", verb="goto", point=[1, 2, 3]), w)
    assert len(goto) == 1 and goto[0].task_type == "goto_waypoint"

    follow = htn.decompose(StructuredIntent(intent_id="i2", verb="track", target_track_id=7), w)
    assert follow[0].task_type == "follow_track"
    assert follow[0].required_capability == "camera"  # 007 capability-awareness

    patrol = htn.decompose(
        StructuredIntent(intent_id="i3", verb="patrol", area=[[0, 0], [10, 0], [10, 10], [0, 10]]),
        w,
    )
    assert len(patrol) == 2  # 4 edges grouped for 2 available drones
    assert all(t.task_type == "patrol_leg" for t in patrol)


def test_htn_errors_are_actionable() -> None:
    htn = HTNDecomposer()
    w = make_world()
    with pytest.raises(ValueError, match="unknown track"):
        htn.decompose(StructuredIntent(intent_id="x", verb="track", target_track_id=99), w)
    with pytest.raises(NotImplementedError, match="CoveragePlanner"):
        htn.decompose(
            StructuredIntent(intent_id="x", verb="scan", area=[[0, 0], [1, 0], [1, 1]]), w
        )


def test_voronoi_scan_partitions_and_sweeps() -> None:
    from airtight.swarm.autonomy.coverage import VoronoiCoverage

    w = make_world(n_drones=2)  # seeds at x=0 and x=5
    htn = HTNDecomposer(coverage=VoronoiCoverage(grid_step=1.0))
    tasks = htn.decompose(
        StructuredIntent(intent_id="s1", verb="scan", area=[[-2, -2], [8, -2], [8, 4], [-2, 4]]), w
    )
    assert len(tasks) == 2 and all(t.task_type == "sweep_cell" for t in tasks)
    # partition property: every waypoint is nearer its own seed than the other's
    seeds = {0: [0.0, 0.0], 1: [5.0, 0.0]}
    all_wps = []
    for k, t in enumerate(tasks):
        for x, y, _ in t.waypoints:
            own = (x - seeds[k][0]) ** 2 + (y - seeds[k][1]) ** 2
            other = (x - seeds[1 - k][0]) ** 2 + (y - seeds[1 - k][1]) ** 2
            assert own <= other + 1e-9
            all_wps.append((x, y))
    assert len(all_wps) == len(set(all_wps))  # cells are disjoint
    # full pipeline: coverage tasks allocate cleanly
    alloc = CBBAAllocator().allocate(tasks, list(w.drones.values()))
    assigned = [t.task_id for p in alloc.values() for t in p]
    assert sorted(assigned) == sorted(t.task_id for t in tasks)
    with pytest.raises(NotImplementedError):
        LLMDecomposer().decompose(StructuredIntent(intent_id="x", verb="goto", point=[0, 0, 0]), w)


# -------------------------------------------------------------------------- bt
def test_bt_sequence_fallback_running() -> None:
    calls: list[str] = []

    def ok(name):
        return Action(name, lambda ctx: (calls.append(name), Status.SUCCESS)[1])

    def fail(name):
        return Action(name, lambda ctx: (calls.append(name), Status.FAILURE)[1])

    assert Sequence(ok("a"), ok("b")).tick({}) == Status.SUCCESS
    assert calls == ["a", "b"]

    calls.clear()
    assert Sequence(ok("a"), fail("b"), ok("c")).tick({}) == Status.FAILURE
    assert calls == ["a", "b"]  # fail-fast: c never ticked

    calls.clear()
    assert Fallback(fail("a"), ok("b")).tick({}) == Status.SUCCESS
    assert calls == ["a", "b"]  # recovery: b tried after a failed

    # RUNNING memory: sequence resumes at the running child, not from the top
    state = {"n": 0}

    def eventually(ctx):
        state["n"] += 1
        return Status.SUCCESS if state["n"] >= 3 else Status.RUNNING

    calls.clear()
    seq = Sequence(ok("setup"), Action("wait", eventually))
    assert seq.tick({}) == Status.RUNNING
    assert seq.tick({}) == Status.RUNNING
    assert seq.tick({}) == Status.SUCCESS
    assert calls == ["setup"]  # setup ran once, not re-ticked every tick


# ------------------------------------------------------------------------ cbba
def test_cbba_proximity_and_conflict_free() -> None:
    drones = [
        DroneState(drone_id="d0", position=[0, 0, 2], capabilities=frozenset({"camera"})),
        DroneState(drone_id="d1", position=[50, 0, 2], capabilities=frozenset({"camera"})),
    ]
    tasks = [
        Task(task_id=f"t{i}", intent_id="i", task_type="goto_waypoint", waypoints=[[x, 0, 2]])
        for i, x in enumerate([1, 3, 49, 51])
    ]
    alloc = CBBAAllocator().allocate(tasks, drones)
    assigned = [t.task_id for path in alloc.values() for t in path]
    assert sorted(assigned) == ["t0", "t1", "t2", "t3"]  # all tasks, no dupes
    assert {t.task_id for t in alloc["d0"]} == {"t0", "t1"}  # near tasks -> near drone
    assert {t.task_id for t in alloc["d1"]} == {"t2", "t3"}


def test_cbba_capability_constraint() -> None:
    drones = [
        DroneState(drone_id="cam", position=[0, 0, 2], capabilities=frozenset({"camera"})),
        DroneState(drone_id="radar_only", position=[1, 0, 2], capabilities=frozenset({"radar"})),
    ]
    tasks = [
        Task(
            task_id="t_follow",
            intent_id="i",
            task_type="follow_track",
            waypoints=[[1, 1, 2]],
            required_capability="camera",
        )
    ]
    alloc = CBBAAllocator().allocate(tasks, drones)
    assert [t.task_id for t in alloc["cam"]] == ["t_follow"]
    assert alloc["radar_only"] == []


# -------------------------------------------------------------- full pipeline
def test_intent_to_allocation_pipeline() -> None:
    w = make_world(n_drones=2)
    htn = HTNDecomposer()
    tasks = htn.decompose(
        StructuredIntent(intent_id="p1", verb="patrol", area=[[0, 0], [20, 0], [20, 20], [0, 20]]),
        w,
    )
    tasks += htn.decompose(StructuredIntent(intent_id="tr1", verb="track", target_track_id=7), w)
    alloc = CBBAAllocator().allocate(tasks, list(w.drones.values()))
    assigned = [t.task_id for path in alloc.values() for t in path]
    assert len(assigned) == len(tasks) == 3  # 2 patrol legs + 1 follow
    assert len(set(assigned)) == 3  # conflict-free
