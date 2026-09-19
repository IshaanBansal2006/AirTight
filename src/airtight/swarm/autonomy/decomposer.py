"""Task decomposition.

`Decomposer` is the swappable interface — THE seam where the Step-3 LLM parser
and Step-5 learned layer later inject decompositions while the HTN degrades to a baseline.
`HTNDecomposer` is the classical implementation: a method library keyed by
intent verb, each method recursively expanding an abstract intent into
primitive `Task`s (things a single drone's behavior tree can execute).

The "scan" verb is BLOCKED on the coverage planner choice (coverage planner: boustrophedon vs
Voronoi vs frontier) — implementing a sweep here would silently decide it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from airtight.swarm.autonomy.world_state import WorldState
    from airtight.swarm.schemas import StructuredIntent

_task_counter = itertools.count()


@dataclass
class Task:
    """A primitive, allocatable unit of work (pre-assignment: no drone yet)."""

    task_id: str
    intent_id: str
    task_type: str  # "goto_waypoint" | "patrol_leg" | "follow_track"
    waypoints: list[list[float]] = field(default_factory=list)
    target_track_id: int | None = None
    required_capability: str | None = None  # capability-aware allocation
    priority: int = 0
    reward: float = 1.0  # base value for the CBBA score


def _new_id(prefix: str) -> str:
    return f"{prefix}-{next(_task_counter)}"


class Decomposer(Protocol):
    """The 020 seam: HTN today; LLMDecomposer (Step 3/5) plugs in here."""

    def decompose(self, intent: StructuredIntent, world: WorldState) -> list[Task]: ...


class HTNDecomposer:
    """Hierarchical decomposition via a verb-keyed method library.

    Each method reads the world state (legitimate HTN practice: methods branch
    on state, e.g. fleet size decides patrol segmentation) and returns
    primitive tasks. First-applicable-method semantics — no backtracking search
    until a mission type needs it (scope guard).
    """

    PATROL_ALTITUDE = 2.0  # m; flight-level default for ground-station ops

    def __init__(self, coverage: object | None = None) -> None:
        # Injected CoveragePlanner; None keeps scan blocked with
        # an actionable error. Import-free typing avoids a module cycle.
        self.coverage = coverage

    def decompose(self, intent: StructuredIntent, world: WorldState) -> list[Task]:
        method = getattr(self, f"_method_{intent.verb}", None)
        if method is None:
            raise ValueError(f"no HTN method for verb '{intent.verb}'")
        return method(intent, world)

    # ------------------------------------------------------------- methods
    def _method_goto(self, intent: StructuredIntent, world: WorldState) -> list[Task]:
        if intent.point is None:
            raise ValueError("goto intent requires 'point' — got none")
        return [
            Task(
                task_id=_new_id("goto"),
                intent_id=intent.intent_id,
                task_type="goto_waypoint",
                waypoints=[intent.point],
                priority=intent.priority,
            )
        ]

    def _method_track(self, intent: StructuredIntent, world: WorldState) -> list[Task]:
        if intent.target_track_id is None:
            raise ValueError("track intent requires 'target_track_id' — got none")
        if intent.target_track_id not in world.tracks:
            raise ValueError(
                f"track intent targets unknown track {intent.target_track_id}; "
                f"known: {sorted(world.tracks)}"
            )
        # Following a target needs an exteroceptive sensor on board (007).
        return [
            Task(
                task_id=_new_id("follow"),
                intent_id=intent.intent_id,
                task_type="follow_track",
                target_track_id=intent.target_track_id,
                required_capability="camera",
                priority=intent.priority,
                reward=2.0,
            )
        ]  # tracking outranks patrolling by default

    def _method_patrol(self, intent: StructuredIntent, world: WorldState) -> list[Task]:
        """Perimeter patrol: polygon boundary -> per-edge leg tasks.

        Segmentation reads fleet size (HTN state-dependent branching): legs are
        grouped so ~each available drone can own a contiguous stretch. This is
        a PERIMETER circuit, not area coverage — area sweeps come later.
        """
        if not intent.area or len(intent.area) < 3:
            raise ValueError("patrol intent requires 'area' polygon (>=3 vertices)")
        verts = [[float(x), float(y), self.PATROL_ALTITUDE] for x, y in intent.area]
        edges = [[verts[i], verts[(i + 1) % len(verts)]] for i in range(len(verts))]
        n_groups = max(1, min(len(world.available_drones()) or 1, len(edges)))
        groups = np.array_split(np.arange(len(edges)), n_groups)
        tasks = []
        for g in groups:
            wps = [edges[i][0] for i in g] + [edges[g[-1]][1]]
            tasks.append(
                Task(
                    task_id=_new_id("patrol"),
                    intent_id=intent.intent_id,
                    task_type="patrol_leg",
                    waypoints=wps,
                    priority=intent.priority,
                )
            )
        return tasks

    def _method_scan(self, intent: StructuredIntent, world: WorldState) -> list[Task]:
        """Area coverage — delegated to the injected 022 planner (Voronoi now;
        FrontierCoverage swaps in for unknown maps on hardware later)."""
        if self.coverage is None:
            raise NotImplementedError(
                "scan requires a CoveragePlanner — construct "
                "HTNDecomposer(coverage=VoronoiCoverage())."
            )
        return self.coverage.plan(intent, world)


class LLMDecomposer:
    """The reserved LLM slot. Filled at Step 3 (language) /
    Step 5 (learned layer); until then it exists so call sites and tests bind to the
    interface, not to HTN."""

    def decompose(self, intent: StructuredIntent, world: WorldState) -> list[Task]:
        raise NotImplementedError("LLMDecomposer arrives with Step 3/5")
