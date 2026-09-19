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
        self._rngs: dict[str, np.random.Generator] = {}

    def rng_for(self, agent_id: str) -> np.random.Generator:
        if agent_id not in self._rngs:
            self._rngs[agent_id] = patrol_rng(self.seed, agent_id)
        return self._rngs[agent_id]

    def staleness(self, t: float) -> Array:
        """max(t - last_seen, 0) where weight > 0, and 0 elsewhere."""
        stale: Array = np.where(self.weight > 0, np.maximum(t - self.last_seen, 0.0), 0.0)
        return stale

    def mark_seen(self, observers: Sequence[Observer], t: float) -> None:
        """Any active observer marks cells seen, fixed sensors included. Only agents retarget."""
        for agent in observers:
            if not agent.active:
                continue
            distance = np.hypot(
                self._centres[..., 0] - agent.pos[0], self._centres[..., 1] - agent.pos[1]
            )
            seen = distance <= agent.footprint_radius_m
            seen &= in_wedge(agent.pos, agent.heading, agent.fov_deg, self._centres)
            self.last_seen[seen] = t

    def retarget(self, agents: list[AgentState], t: float) -> None:
        active = sorted((a for a in agents if a.active), key=lambda a: a.index)
        for agent in active:
            to_target = float(np.linalg.norm(agent.target - agent.pos))
            arrived = to_target <= agent.footprint_radius_m / 2.0
            if not arrived and t - agent.last_retarget_t < self.retarget_period_s:
                continue
            peers = {a.index: a.pos for a in active if a is not agent}
            region = voronoi_mask(self._centres, agent.pos, peers, agent.index)
            distance = np.hypot(
                self._centres[..., 0] - agent.pos[0], self._centres[..., 1] - agent.pos[1]
            )
            base = self.staleness(t) * self.weight / (1.0 + distance / self.d0_m)
            utility = base * region
            if not np.any(utility > 0):
                utility = base
            if not np.any(utility > 0):
                continue  # nothing worth visiting: keep the old target, draw nothing
            flat = utility.ravel()
            positive = np.flatnonzero(flat > 0)
            k = max(1, int(self.top_fraction * len(positive)))
            # stable sort: ties resolve by cell index, so the candidate set is reproducible
            top = positive[np.argsort(-flat[positive], kind="stable")][:k]
            pick = int(top[int(self.rng_for(agent.agent_id).integers(k))])
            row, col = divmod(pick, self.grid.cols)
            agent.target = self._centres[row, col].copy()
            agent.last_retarget_t = t


def step_agents(agents: list[AgentState], dt: float) -> None:
    """Each active agent moves toward its target by at most speed_mps * dt, never past it."""
    for agent in agents:
        if not agent.active:
            continue
        delta = agent.target - agent.pos
        distance = float(np.linalg.norm(delta))
        if distance == 0.0:
            continue
        agent.heading = math.atan2(float(delta[1]), float(delta[0]))
        step = agent.speed_mps * dt
        if distance <= step + _ARRIVE_EPS_M:
            agent.pos = agent.target.copy()
        else:
            agent.pos = agent.pos + delta * (step / distance)
