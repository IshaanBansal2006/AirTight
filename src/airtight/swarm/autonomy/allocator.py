"""CBBA task allocation.

Two alternating phases per round:
  1. BUNDLE BUILD (greedy, per drone): repeatedly add the task with the best
     *marginal* score improvement at its best path-insertion point, but only if
     that score would OUTBID the task's current global winner.
  2. CONSENSUS: drones exchange (bids, winners); for each task the highest bid
     wins (ties -> lower drone_id). A drone that loses a task drops it AND every
     task it added later (bundle truncation — later additions were scored
     assuming the lost task was in the path, so their bids are stale).

Rounds repeat until nothing changes. Scores are time-discounted rewards
S = lambda^tau * reward (tau = arrival time along the drone's path), which are
diminishing-marginal-gain — the property CBBA's convergence guarantee needs —
when tasks are APPENDED. With best-position insertion (what phase 1 does) a
task already in the path can be a paid-for detour into a neighbour's area
and RAISE that neighbour's marginal gain; tests/test_learned_score.py pins a
minority of random cases doing exactly that. The guarantee was therefore
never strict here; the round cap is what ends the auction.

Documented simplification (backlog): fully-connected synchronous communication,
so consensus is a global max per task each round. The full Choi conflict-
resolution table (36 rules with timestamps) is only needed for multi-hop /
asynchronous networks — it slots in behind the same interface when comms get
realistic. Convergence bound: <= N_tasks rounds under full connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from airtight.swarm.autonomy.decomposer import Task
    from airtight.swarm.autonomy.world_state import DroneState

LAMBDA = 0.95  # time discount per second of travel; sets urgency of proximity
MAX_BUNDLE = 5  # tasks per drone per allocation round (compute guard)


@dataclass
class _Agent:
    """Per-drone CBBA working state."""

    drone: DroneState
    path: list[Task] = field(default_factory=list)  # ordered execution route


def _task_entry(task: Task) -> np.ndarray:
    return np.asarray(task.waypoints[0] if task.waypoints else [0.0, 0.0, 0.0])


def _task_exit(task: Task) -> np.ndarray:
    return np.asarray(task.waypoints[-1] if task.waypoints else [0.0, 0.0, 0.0])


def _path_score(agent: _Agent, path: list[Task]) -> float:
    """Sum of lambda^arrival_time * reward along the path (straight-line legs)."""
    pos = np.asarray(agent.drone.position)
    t = 0.0
    score = 0.0
    for task in path:
        t += float(np.linalg.norm(_task_entry(task) - pos)) / agent.drone.speed
        score += (LAMBDA**t) * task.reward
        # traverse the task's own waypoints before heading onward
        wps = [np.asarray(w) for w in task.waypoints]
        for a, b in zip(wps, wps[1:], strict=False):
            t += float(np.linalg.norm(b - a)) / agent.drone.speed
        pos = _task_exit(task)
    return score


def _capable(drone: DroneState, task: Task) -> bool:
    return task.required_capability is None or task.required_capability in drone.capabilities


class PathScore(Protocol):
    """The 023 seam: what a drone bids for an ordered path of tasks."""

    def score(self, drone: DroneState, path: list[Task]) -> float: ...


class TimeDiscountedScore:
    """The hand-written score: sum of lambda^arrival * reward. Diminishing
    marginal gain by construction, which is what CBBA's convergence proof needs."""

    def score(self, drone: DroneState, path: list[Task]) -> float:
        return _path_score(_Agent(drone=drone), path)


class CBBAAllocator:
    """Bundle build + consensus (021). The SCORE is pluggable (023): the
    default is the time-discounted hand-written one; a learned score may be
    substituted per drone. With a score that is not diminishing-marginal-gain
    the convergence proof no longer applies — `last_converged` records whether
    the auction settled before the round cap, so a caller can measure how
    often it did."""

    def __init__(
        self, score: PathScore | None = None, score_overrides: dict[str, PathScore] | None = None
    ) -> None:
        self.score = score if score is not None else TimeDiscountedScore()
        self.score_overrides = dict(score_overrides or {})
        self.last_rounds = 0
        self.last_converged = True

    def score_for(self, drone_id: str) -> PathScore:
        return self.score_overrides.get(drone_id, self.score)

    def allocate(
        self, tasks: list[Task], drones: list[DroneState], max_rounds: int | None = None
    ) -> dict[str, list[Task]]:
        """Returns drone_id -> ordered task path. Unallocatable tasks are absent
        (no capable drone, or bundles full) — caller decides whether that's an
        error or a next-round situation."""
        agents = {d.drone_id: _Agent(drone=d) for d in drones if d.available}
        if not agents or not tasks:
            return {a: [] for a in agents}
        bids: dict[str, float] = {t.task_id: 0.0 for t in tasks}
        winners: dict[str, str | None] = {t.task_id: None for t in tasks}
        {t.task_id: t for t in tasks}
        cap = max_rounds or len(tasks) + 1
        self.last_converged = False

        for rnd in range(cap):
            self.last_rounds = rnd + 1
            changed = False
            # -- phase 1: bundle building --------------------------------
            for aid in sorted(agents):
                agent = agents[aid]
                scorer = self.score_for(aid)
                while len(agent.path) < MAX_BUNDLE:
                    best = None  # (marginal, insert_pos, task)
                    base = scorer.score(agent.drone, agent.path)
                    for task in tasks:
                        if task in agent.path or not _capable(agent.drone, task):
                            continue
                        for pos in range(len(agent.path) + 1):
                            trial = agent.path[:pos] + [task] + agent.path[pos:]
                            marginal = scorer.score(agent.drone, trial) - base
                            if marginal > bids[task.task_id] + 1e-12 and (
                                best is None or marginal > best[0]
                            ):
                                best = (marginal, pos, task)
                    if best is None:
                        break
                    marginal, pos, task = best
                    agent.path.insert(pos, task)
                    bids[task.task_id] = marginal
                    winners[task.task_id] = aid
                    changed = True
            # -- phase 2: consensus (fully connected: global max wins) ---
            for aid in sorted(agents):
                agent = agents[aid]
                for i, task in enumerate(agent.path):
                    if winners[task.task_id] != aid:
                        # lost this task -> drop it and everything added after
                        for dropped in agent.path[i:]:
                            if winners[dropped.task_id] == aid:
                                bids[dropped.task_id] = 0.0
                                winners[dropped.task_id] = None
                        agent.path = agent.path[:i]
                        changed = True
                        break
            if not changed:
                self.last_converged = True
                break

        return {aid: list(agents[aid].path) for aid in agents}
