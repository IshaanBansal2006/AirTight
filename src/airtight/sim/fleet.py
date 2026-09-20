"""Patrolling agents: state, the randomized stalest-cell patrol controller, and motion.

Each agent heads for a cell chosen at random among the most useful cells of its own Voronoi
region, where useful means stale, weighted and near. There is deliberately no Lloyd or other
centroid-seeking controller here: once staleness saturates such a controller parks the agents.

Patrol target choice is one random stream per agent,
default_rng([seed, 4, crc32(agent_id) % 1000000]), so an agent's choices never depend on how
many other agents exist or on iteration order.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.sim import adapt
from airtight.sim.geometry import in_wedge, voronoi_mask
from airtight.sim.sensing import FixedObserver

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from airtight.contracts import FleetConfig, SensorCurves, Site
    from airtight.sim.geometry import Grid
    from airtight.sim.sensing import Observer

    Array = npt.NDArray[np.float64]

PATROL_STREAM = 4
_ARRIVE_EPS_M = 1e-9


@dataclass
class AgentState:
    agent_id: str
    index: int  # position in the fleet; also the agent's id in Voronoi tie-breaks
    pos: Array  # (2,) metres
    heading: float  # radians
    target: Array  # (2,) metres
    speed_mps: float
    footprint_radius_m: float
    fov_deg: float
    active: bool = True
    last_retarget_t: float = -math.inf
    sensor_type: str = ""  # key into the sensor curves; adapt raises a clear KeyError on ""
    mode: str = "patrol"  # patrol, returning or charging; see battery.py. active is False
    # exactly when charging, so a returning agent still senses.
    home: Array | None = None  # its own pad; None for agents built without one

    @property
    def patrolling(self) -> bool:
        """Takes patrol targets and counts as a Voronoi peer. A returning agent does neither:
        it still sees, but its neighbours must already be covering its area."""
        return self.active and self.mode == "patrol"


def make_agents(site: Site, fleet: FleetConfig, sensor_curves: SensorCurves) -> list[AgentState]:
    agents = []
    for index, agent_id in enumerate(adapt.agent_ids(fleet)):
        sensor_type = adapt.agent_sensor_type(fleet, agent_id)
        pos = adapt.start_position(site, fleet, agent_id)
        agents.append(
            AgentState(
                agent_id=agent_id,
                index=index,
                pos=pos,
                heading=0.0,
                target=pos.copy(),
                speed_mps=adapt.agent_speed_mps(fleet, agent_id),
                footprint_radius_m=adapt.sensor_footprint_radius_m(sensor_curves, sensor_type),
                fov_deg=adapt.sensor_fov_deg(sensor_curves, sensor_type),
                sensor_type=sensor_type,
                home=pos.copy(),
            )
        )
    return agents


def patrol_rng(seed: int, agent_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, PATROL_STREAM, zlib.crc32(agent_id.encode()) % 1000000])


class PatrolController:
    def __init__(
        self,
        grid: Grid,
        weight: Array,
        seed: int,
        t_start: float,
        retarget_period_s: float = 10.0,
        d0_m: float = 60.0,
        top_fraction: float = 0.05,
        stale_init_s: float = 300.0,
    ) -> None:
        if weight.shape != grid.shape:
            raise ValueError(f"weight shape {weight.shape} does not match grid shape {grid.shape}")
        self.grid = grid
        self.weight = weight
        self.seed = seed
        self.retarget_period_s = retarget_period_s
        self.d0_m = d0_m
        self.top_fraction = top_fraction
        # Staleness is never negative however long the warm-up is.
        self.last_seen: Array = np.full(grid.shape, t_start - stale_init_s, dtype=np.float64)
        self._centres = grid.cell_centers()
        self._weighted = weight > 0
        self._rngs: dict[str, np.random.Generator] = {}
        # Fixed sensors never move: the disk+wedge mask is constant for the episode.
        self._static_footprint: dict[int, tuple[int, int, int, int, np.ndarray]] = {}
        self._xmin = grid.xmin
        self._ymin = grid.ymin
        self._cell_size = grid.cell_size
        self._rows = grid.rows
        self._cols = grid.cols
        self._work_dx = np.empty(grid.shape, dtype=np.float64)
        self._work_dy = np.empty(grid.shape, dtype=np.float64)
        self._work_dist = np.empty(grid.shape, dtype=np.float64)
        self._work_stale = np.empty(grid.shape, dtype=np.float64)
        self._work_base = np.empty(grid.shape, dtype=np.float64)
        self._work_util = np.empty(grid.shape, dtype=np.float64)
        self._empty_seen = np.zeros((0, 0), dtype=np.bool_)

    def rng_for(self, agent_id: str) -> np.random.Generator:
        if agent_id not in self._rngs:
            self._rngs[agent_id] = patrol_rng(self.seed, agent_id)
        return self._rngs[agent_id]

    def staleness(self, t: float) -> Array:
        """max(t - last_seen, 0) where weight > 0, and 0 elsewhere."""
        stale: Array = np.zeros(self.grid.shape, dtype=np.float64)
        stale[self._weighted] = np.maximum(t - self.last_seen[self._weighted], 0.0)
        return stale

    def weighted_staleness(self, t: float) -> Array:
        """1-D staleness of cells with positive patrol weight. Same values as staleness(t)[weight>0]."""
        vals: Array = np.maximum(t - self.last_seen[self._weighted], 0.0)
        return vals

    def _footprint_window(
        self, pos: Array, heading: float, fov_deg: float, radius_m: float
    ) -> tuple[int, int, int, int, np.ndarray]:
        """Boolean mask of cell centres inside the observer's disk and wedge, plus its bbox.

        A 360° camera (the drone) is a disk: the wedge is identically true and is not computed.
        """
        x, y = float(pos[0]), float(pos[1])
        cs = self._cell_size
        col0 = max(0, int(math.floor((x - radius_m - self._xmin) / cs)) - 1)
        col1 = min(self._cols, int(math.ceil((x + radius_m - self._xmin) / cs)) + 1)
        row0 = max(0, int(math.floor((y - radius_m - self._ymin) / cs)) - 1)
        row1 = min(self._rows, int(math.ceil((y + radius_m - self._ymin) / cs)) + 1)
        if row0 >= row1 or col0 >= col1:
            return row0, row1, col0, col1, self._empty_seen
        window = self._centres[row0:row1, col0:col1]
        seen = np.hypot(window[..., 0] - x, window[..., 1] - y) <= radius_m
        if fov_deg < 360.0:
            seen &= in_wedge(pos, heading, fov_deg, window)
        return row0, row1, col0, col1, seen

    def mark_seen(self, observers: Sequence[Observer], t: float) -> None:
        """Any active observer marks cells seen, fixed sensors included. Only agents retarget.

        Only cells whose centre can sit inside the footprint disk are tested. The disk test and
        the wedge test are unchanged, so the set of marked cells is identical to a full-grid pass.
        Fixed sensors reuse a cached mask: their pose never changes.
        """
        for agent in observers:
            if not agent.active:
                continue
            if isinstance(agent, FixedObserver):
                cached = self._static_footprint.get(id(agent))
                if cached is None:
                    row0, row1, col0, col1, seen = self._footprint_window(
                        agent.pos, agent.heading, agent.fov_deg, agent.footprint_radius_m
                    )
                    cached = (row0, row1, col0, col1, seen)
                    self._static_footprint[id(agent)] = cached
                row0, row1, col0, col1, seen = cached
            else:
                row0, row1, col0, col1, seen = self._footprint_window(
                    agent.pos, agent.heading, agent.fov_deg, agent.footprint_radius_m
                )
            if seen.size:
                self.last_seen[row0:row1, col0:col1][seen] = t

    def _due_to_retarget(self, agent: AgentState, t: float) -> bool:
        dx = float(agent.target[0] - agent.pos[0])
        dy = float(agent.target[1] - agent.pos[1])
        arrived = math.hypot(dx, dy) <= agent.footprint_radius_m / 2.0
        return arrived or t - agent.last_retarget_t >= self.retarget_period_s

    def retarget(self, agents: list[AgentState], t: float) -> None:
        due_ids = {a.agent_id for a in agents if a.patrolling and self._due_to_retarget(a, t)}
        if not due_ids:
            return
        active = sorted((a for a in agents if a.patrolling), key=lambda a: a.index)
        weighted_stale: Array | None = None
        for agent in active:
            if agent.agent_id not in due_ids:
                continue
            peers = {a.index: a.pos for a in active if a is not agent}
            region = voronoi_mask(self._centres, agent.pos, peers, agent.index)
            np.subtract(self._centres[..., 0], agent.pos[0], out=self._work_dx)
            np.subtract(self._centres[..., 1], agent.pos[1], out=self._work_dy)
            np.hypot(self._work_dx, self._work_dy, out=self._work_dist)
            if weighted_stale is None:
                np.subtract(t, self.last_seen, out=self._work_stale)
                np.maximum(self._work_stale, 0.0, out=self._work_stale)
                np.multiply(self._work_stale, self.weight, out=self._work_stale)
                weighted_stale = self._work_stale
            np.divide(weighted_stale, 1.0 + self._work_dist / self.d0_m, out=self._work_base)
            np.multiply(self._work_base, region, out=self._work_util)
            utility: Array = self._work_util
            if not np.any(utility > 0):
                utility = self._work_base
            if not np.any(utility > 0):
                continue  # nothing worth visiting: keep the old target, draw nothing
            flat = utility.ravel()
            positive = np.flatnonzero(flat > 0)
            k = max(1, int(self.top_fraction * len(positive)))
            # stable sort: ties resolve by cell index, so the candidate set is reproducible
            top = positive[np.argsort(-flat[positive], kind="stable")][:k]
            pick = int(top[int(self.rng_for(agent.agent_id).integers(k))])
            row, col = divmod(pick, self._cols)
            agent.target = self._centres[row, col].copy()
            agent.last_retarget_t = t


def step_agents(agents: list[AgentState], dt: float) -> None:
    """Each active agent moves toward its target by at most speed_mps * dt, never past it."""
    for agent in agents:
        if not agent.active:
            continue
        dx = float(agent.target[0]) - float(agent.pos[0])
        dy = float(agent.target[1]) - float(agent.pos[1])
        distance = math.hypot(dx, dy)
        if distance == 0.0:
            continue
        agent.heading = math.atan2(dy, dx)
        step = agent.speed_mps * dt
        if distance <= step + _ARRIVE_EPS_M:
            agent.pos[0] = float(agent.target[0])
            agent.pos[1] = float(agent.target[1])
        else:
            s = step / distance
            agent.pos[0] += dx * s
            agent.pos[1] += dy * s
