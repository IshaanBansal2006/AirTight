"""The battery clock: when each agent is on duty, heading home, or charging.

Definitions follow the contract and lane C: an agent's cycle is endurance_s + charge_time_s. It
is on duty while its cycle position u is below endurance_s and charging for the charge_time_s
after that. u = (t_abs + offset_s) mod cycle, where t_abs is absolute time and offset_s is the
contract's stagger offset. At t_abs = 0 an agent with offset 0 has just left its pad fully
charged.

Modes are driven by that clock, never by integrating energy, so the cycle is exactly periodic
and the state at any absolute time is closed form. No long fleet run or cached snapshot is
needed to start an episode at a given phase.

An agent heads home when the time it has left on duty equals its travel time home, so it
reaches its pad as its endurance runs out. The contract's threshold_frac reserve is the safety
margin for that flight and is not modelled further. If an agent is still short of its pad when
charging starts (it can be, by a step or two), it is placed on the pad.

Every agent has its own pad, the point adapt.start_position gives it. Dock capacity is ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from airtight.sim import adapt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from airtight.contracts import FleetConfig
    from airtight.sim.fleet import AgentState

PATROL = "patrol"
RETURNING = "returning"
CHARGING = "charging"


@dataclass(frozen=True)
class BatteryClock:
    endurance_s: float
    charge_time_s: float
    offset_s: float = 0.0
    cycle_s: float = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_s", self.endurance_s + self.charge_time_s)

    def position_s(self, t_abs: float) -> float:
        """u in [0, cycle_s): how far into its cycle the agent is. Fine for negative t_abs."""
        return (t_abs + self.offset_s) % self.cycle_s

    def charging(self, t_abs: float) -> bool:
        return self.position_s(t_abs) >= self.endurance_s

    def on_duty_left_s(self, t_abs: float) -> float:
        """Seconds until charging starts; 0.0 while charging."""
        return max(self.endurance_s - self.position_s(t_abs), 0.0)

    def charge_left_s(self, t_abs: float) -> float:
        """Seconds until the agent is back on duty; 0.0 while on duty."""
        u = self.position_s(t_abs)
        return self.cycle_s - u if u >= self.endurance_s else 0.0


class ModeChange(NamedTuple):
    agent_id: str
    old: str
    new: str


def make_clocks(fleet: FleetConfig) -> dict[str, BatteryClock]:
    """One clock per agent that charges. Agents that never charge have no entry."""
    clocks = {}
    for agent_id in adapt.agent_ids(fleet):
        spec = adapt.agent_energy(fleet, agent_id)
        if spec is not None:
            clocks[agent_id] = BatteryClock(spec.endurance_s, spec.charge_time_s, spec.offset_s)
    return clocks


def step_battery(
    agents: Sequence[AgentState], clocks: Mapping[str, BatteryClock], t_abs: float, dt: float
) -> list[ModeChange]:
    """Bring every agent's mode in line with its clock at absolute time t_abs.

    Safe to call at any time, including the first step of an episode: the mode depends only on
    the clock and on where the agent is, not on what was called before.
    """
    changes = []
    for agent in agents:
        clock = clocks.get(agent.agent_id)
        if clock is None or agent.home is None:
            continue
        old = agent.mode
        if clock.charging(t_abs):
            if old != CHARGING:
                agent.mode = CHARGING
                agent.active = False
                agent.pos = agent.home.copy()
                agent.target = agent.home.copy()
        elif old == CHARGING:
            agent.mode = PATROL
            agent.active = True
            agent.last_retarget_t = -math.inf  # pick a patrol target at once
        elif old == PATROL:
            travel_s = float(np.linalg.norm(agent.home - agent.pos)) / agent.speed_mps
            if clock.on_duty_left_s(t_abs) <= travel_s + dt:
                agent.mode = RETURNING
                agent.target = agent.home.copy()
        if agent.mode != old:
            changes.append(ModeChange(agent.agent_id, old, agent.mode))
    return changes
