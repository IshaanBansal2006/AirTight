"""Step-1/2 execution + safety tests: BT leaves on the KinematicBackend
(mission-in-miniature, no ROS) and the L4 gate's fail-safe state machine."""

from __future__ import annotations

import numpy as np
import pytest

from airtight.swarm.autonomy.allocator import CBBAAllocator
from airtight.swarm.autonomy.bt import Status
from airtight.swarm.autonomy.decomposer import HTNDecomposer, Task
from airtight.swarm.autonomy.executor import ARRIVE_TOL, Executor, KinematicBackend, SmoothBackend
from airtight.swarm.autonomy.world_state import DroneState, WorldState
from airtight.swarm.edge import rotation
from airtight.swarm.hol.gate import ApprovalGate
from airtight.swarm.schemas import EngagementProposal, StructuredIntent, TrackMsg


def make_world() -> WorldState:
    w = WorldState()
    w.update_drone(
        DroneState(drone_id="d0", position=[0.0, 0.0, 2.0], capabilities=frozenset({"camera"}))
    )
    w.update_drone(
        DroneState(drone_id="d1", position=[10.0, 0.0, 2.0], capabilities=frozenset({"camera"}))
    )
    return w


def run_until_idle(ex: Executor, max_ticks: int = 2000) -> int:
    for i in range(max_ticks):
        ex.tick(0.1)
        if ex.idle():
            return i
    pytest.fail("executor never went idle")


def test_mission_in_miniature_patrol() -> None:
    """intent -> HTN -> CBBA -> BT execution -> drones physically fly the legs."""
    w = make_world()
    tasks = HTNDecomposer().decompose(
        StructuredIntent(intent_id="p", verb="patrol", area=[[0, 0], [8, 0], [8, 8], [0, 8]]), w
    )
    alloc = CBBAAllocator().allocate(tasks, list(w.drones.values()))
    ex = Executor(KinematicBackend(w), w)
    ex.assign(alloc)
    run_until_idle(ex)
    assert sorted(ex.completed) == sorted(t.task_id for t in tasks)
    assert not ex.failed
    # each drone physically ended at ITS final leg waypoint
    for drone_id, path in alloc.items():
        if path:
            final = np.asarray(path[-1].waypoints[-1])
            assert (
                np.linalg.norm(np.asarray(w.drones[drone_id].position) - final) < ARRIVE_TOL + 1e-6
            )


def test_follow_track_chases_and_fails_on_loss() -> None:
    w = make_world()
    w.update_tracks(
        [
            TrackMsg(
                track_id=3,
                timestamp=0.0,
                position=[5, 5, 0.5],
                velocity=[0, 0, 0],
                extent=[0.2] * 3,
                position_sqrt_cov=[0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
            )
        ],
        0.0,
    )
    task = Task(
        task_id="f",
        intent_id="i",
        task_type="follow_track",
        target_track_id=3,
        required_capability="camera",
    )
    ex = Executor(KinematicBackend(w), w)
    ex.assign({"d0": [task]})
    for _ in range(200):
        ex.tick(0.1)
    # chasing: drone closed in on the (static) target's standoff point
    d = np.linalg.norm(np.asarray(w.drones["d0"].position) - np.asarray([5, 5, 2.5]))
    assert d < 0.5
    assert ex.tick(0.1)["d0"] == Status.RUNNING  # following never self-completes
    w.update_tracks([], 1.0)  # track lost
    ex.tick(0.1)
    assert ex.failed == ["f"]  # surfaces as the replan trigger


def test_gate_approve_deny_and_timeout_fail_safe() -> None:
    gate = ApprovalGate()
    p1 = EngagementProposal(proposal_id="p1", action="designate", rationale="t", deadline_s=10)
    p2 = EngagementProposal(proposal_id="p2", action="designate", rationale="t", deadline_s=5)
    gate.submit(p1, now=0.0)
    gate.submit(p2, now=0.0)

    assert gate.decide("p1", approve=True, operator="ishaan", now=1.0) is True
    with pytest.raises(KeyError, match="p1"):  # no double-deciding
        gate.decide("p1", approve=False, operator="ishaan", now=1.1)

    assert gate.tick(now=4.0) == []  # p2 not yet expired
    assert gate.tick(now=5.0) == ["p2"]  # silence never authorizes: auto-DENY
    events = [(e["event"], e["proposal_id"]) for e in gate.audit]
    assert events == [
        ("proposed", "p1"),
        ("proposed", "p2"),
        ("approved", "p1"),
        ("auto_denied_timeout", "p2"),
    ]


def test_smooth_backend_flies_the_patrol_with_bounded_acceleration() -> None:
    """same mission as the kinematic backend, but |dv/dt| <= max_accel
    and the platform carries a heading that follows its velocity."""
    w = make_world()
    tasks = HTNDecomposer().decompose(
        StructuredIntent(intent_id="p", verb="patrol", area=[[0, 0], [8, 0], [8, 8], [0, 8]]), w
    )
    alloc = CBBAAllocator().allocate(tasks, list(w.drones.values()))
    backend = SmoothBackend(w, max_accel=2.0, max_yaw_rate=1.5)
    ex = Executor(backend, w)
    ex.assign(alloc)
    dt = 0.1
    prev_v = {d: np.zeros(3) for d in w.drones}
    steady = dict.fromkeys(w.drones, 0)  # consecutive ticks at speed on a constant course
    prev_course = dict.fromkeys(w.drones, 0.0)
    checked_heading = False
    for _ in range(3000):
        ex.tick(dt)
        for did, d in w.drones.items():
            v = np.asarray(d.velocity)
            assert np.linalg.norm(v - prev_v[did]) <= backend.max_accel * dt + 1e-9 or np.allclose(
                v, 0.0
            )  # arrival snaps to rest (documented)
            prev_v[did] = v
            q = np.asarray(d.orientation)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-9
            # heading follows course through a yaw-rate limit, and a goto leaf
            # hands over to the next leg while the drone is still moving, so the
            # course swings at every corner: only assert once the course has
            # been steady at speed long enough for yaw to have caught up
            course = float(np.arctan2(v[1], v[0]))
            turning = abs(rotation.wrap_angle(course - prev_course[did])) > 0.02
            prev_course[did] = course
            steady[did] = 0 if (turning or np.hypot(v[0], v[1]) < 1.0) else steady[did] + 1
            if steady[did] * dt >= 1.5:
                assert abs(rotation.wrap_angle(rotation.yaw_of(q) - course)) < 0.05
                checked_heading = True
        if ex.idle():
            break
    assert checked_heading
    assert ex.idle() and not ex.failed
    for drone_id, path in alloc.items():
        if path:
            final = np.asarray(path[-1].waypoints[-1])
            assert (
                np.linalg.norm(np.asarray(w.drones[drone_id].position) - final) < ARRIVE_TOL + 1e-6
            )


def test_world_state_attaches_pose_estimates_to_known_drones_only() -> None:
    from airtight.swarm.schemas import DronePoseFrame, DronePoseMsg

    w = make_world()
    frame = DronePoseFrame(
        timestamp=3.0,
        poses=[
            DronePoseMsg(
                drone_id="d0",
                timestamp=3.0,
                position=[0.1, 0, 2],
                orientation=[1, 0, 0, 0],
                pose_sqrt_cov=[0.0] * 36,
            ),
            DronePoseMsg(
                drone_id="ghost", timestamp=3.0, position=[9, 9, 9], orientation=[1, 0, 0, 0]
            ),
        ],
    )
    w.update_pose_estimates(frame)
    assert w.drones["d0"].pose_estimate is not None
    assert w.drones["d0"].pose_estimate.position == [0.1, 0, 2]
    assert w.drones["d1"].pose_estimate is None
    assert "ghost" not in w.drones and w.time == 3.0
