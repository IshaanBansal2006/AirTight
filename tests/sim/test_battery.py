from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from airtight.sim import adapt, scenarios
from airtight.sim import fleet as fleet_module
from airtight.sim.battery import (
    CHARGING,
    PATROL,
    RETURNING,
    BatteryClock,
    make_clocks,
    step_battery,
)
from airtight.sim.fleet import AgentState, PatrolController, make_agents, step_agents
from airtight.sim.geometry import Grid, inside_mask, patrol_weight

if TYPE_CHECKING:
    from airtight.contracts import FleetConfig

DT = 0.25
ENDURANCE, CHARGE = 1500.0, 2100.0  # the yard_night drone
CYCLE = ENDURANCE + CHARGE


def _walk_cycle(clock: BatteryClock, t_abs: float) -> tuple[bool, float]:
    """Brute force: step through the cycle from absolute time 0 in small steps."""
    step = 0.05
    u = clock.offset_s % clock.cycle_s
    for _ in range(round(t_abs / step)):
        u += step
        if u >= clock.cycle_s:
            u -= clock.cycle_s
    return u >= clock.endurance_s, u


def test_cycle_is_endurance_plus_charge_time() -> None:
    clock = BatteryClock(ENDURANCE, CHARGE)
    assert clock.cycle_s == 3600.0
    assert not clock.charging(0.0) and clock.on_duty_left_s(0.0) == ENDURANCE
    assert not clock.charging(1499.9) and clock.charging(1500.0)
    assert clock.charge_left_s(1500.0) == CHARGE and clock.on_duty_left_s(1500.0) == 0.0
    assert clock.charging(3599.9) and not clock.charging(3600.0)
    assert clock.position_s(-10.0) == 3590.0 and clock.charging(-10.0)  # negative time wraps


def test_closed_form_agrees_with_a_brute_force_walk() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        clock = BatteryClock(ENDURANCE, CHARGE, offset_s=float(rng.uniform(0, CYCLE)))
        t_abs = float(rng.uniform(0, 3 * CYCLE))
        charging, u = _walk_cycle(clock, t_abs)
        near_an_edge = min(abs(u - ENDURANCE), u, CYCLE - u) < DT
        if not near_an_edge:
            assert clock.charging(t_abs) == charging
        gap = abs(clock.position_s(t_abs) - u)
        assert min(gap, CYCLE - gap) < DT


def test_exactly_periodic() -> None:
    clock = BatteryClock(ENDURANCE, CHARGE, offset_s=412.5)
    for t in (0.0, 17.25, 1499.0, 1500.0, 2900.75):
        for k in (1, 2, 7):
            assert clock.charging(t + k * CYCLE) == clock.charging(t)
            assert clock.position_s(t + k * CYCLE) == pytest.approx(clock.position_s(t))


def _fleet_run(fleet: FleetConfig, seconds: float) -> dict[str, list[tuple[float, str, float]]]:
    """A fleet-only patrol on yard_night from t_abs = 0. Per agent: (t_abs, new mode, distance
    from home just before the change)."""
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    grid = Grid(*adapt.bounds(site), 5.0)
    weight = patrol_weight(grid, inside_mask(grid, adapt.perimeter(site)), adapt.assets(site))
    agents = make_agents(site, fleet, curves)
    clocks = make_clocks(fleet)
    controller = PatrolController(grid, weight, seed=11, t_start=0.0)
    history: dict[str, list[tuple[float, str, float]]] = {a.agent_id: [] for a in agents}
    for k in range(round(seconds / DT)):
        t = k * DT
        before = {a.agent_id: float(np.linalg.norm(a.pos - a.home)) for a in agents}
        for change in step_battery(agents, clocks, t, DT):
            history[change.agent_id].append((t, change.new, before[change.agent_id]))
        controller.mark_seen(agents, t)
        controller.retarget(agents, t)
        step_agents(agents, DT)
    return history


def test_fleet_run_keeps_the_exact_cycle_and_arrives_home_in_time() -> None:
    history = _fleet_run(scenarios.load_fleet("2drones"), 3 * CYCLE + 1.0)
    for events in history.values():
        modes = [mode for _, mode, _ in events]
        assert modes == [RETURNING, CHARGING, PATROL] * 3
        for k in range(3):
            t_return, t_charge, t_patrol = (events[3 * k + i][0] for i in range(3))
            assert t_charge == k * CYCLE + ENDURANCE  # charging starts on the clock, exactly
            assert t_patrol == (k + 1) * CYCLE  # and lasts charge_time_s exactly
            assert t_return < t_charge and t_charge - t_return < 60.0  # a transit, not a camp-out
            # it had reached its pad, to within a couple of steps of flight, before being docked
            assert events[3 * k + 1][2] <= 2 * 8.0 * DT


def test_returning_agent_still_sees_but_is_no_peer_and_is_not_retargeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = Grid(0.0, 0.0, 300.0, 200.0, 5.0)
    weight = np.ones(grid.shape)
    home = np.array([40.0, 40.0])

    def agent(agent_id: str, index: int, xy: tuple[float, float]) -> AgentState:
        pos = np.array(xy)
        return AgentState(agent_id, index, pos, 0.0, pos.copy(), 8.0, 20.0, 360.0, home=home.copy())

    far, other = agent("d0", 0, (240.0, 40.0)), agent("d1", 1, (150.0, 150.0))
    clocks = {"d0": BatteryClock(ENDURANCE, CHARGE)}
    t_abs = ENDURANCE - 20.0  # 200 m from home at 8 m/s is 25 s: time to go
    assert step_battery([far, other], clocks, t_abs, DT) == [("d0", PATROL, RETURNING)]
    assert far.active and not far.patrolling and far.target.tolist() == [40.0, 40.0]
    assert other.mode == PATROL  # no clock, never changes

    peers_seen: list[tuple[int, set[int]]] = []
    real = fleet_module.voronoi_mask

    def spy(centres: Any, own_xy: Any, peers: dict[int, Any], own_id: int) -> Any:
        peers_seen.append((own_id, set(peers)))
        return real(centres, own_xy, peers, own_id)

    monkeypatch.setattr(fleet_module, "voronoi_mask", spy)
    controller = PatrolController(grid, weight, seed=1, t_start=0.0)
    controller.mark_seen([far, other], 5.0)
    assert controller.last_seen[grid.cell_of(240.0, 40.0)] == 5.0  # still sees
    controller.retarget([far, other], 5.0)
    assert far.target.tolist() == [40.0, 40.0] and far.last_retarget_t == -math.inf
    assert other.last_retarget_t == 5.0
    assert peers_seen == [(1, set())]  # only d1 retargeted, and d0 was not counted as a peer


def test_charging_agent_is_docked_inactive_and_comes_back() -> None:
    home = np.array([40.0, 40.0])
    pos = np.array([44.0, 40.0])
    agent = AgentState("d0", 0, pos, 0.0, pos.copy(), 8.0, 20.0, 360.0, home=home)
    clocks = {"d0": BatteryClock(ENDURANCE, CHARGE)}
    step_battery([agent], clocks, ENDURANCE + 1.0, DT)
    assert agent.mode == CHARGING and not agent.active and agent.pos.tolist() == [40.0, 40.0]
    assert step_battery([agent], clocks, ENDURANCE + 500.0, DT) == []
    agent.last_retarget_t = 99.0
    assert step_battery([agent], clocks, CYCLE + 1.0, DT) == [("d0", CHARGING, PATROL)]
    assert agent.active and agent.patrolling and agent.last_retarget_t == -math.inf


def test_synchronized_and_staggered_overlap_matches_the_arithmetic() -> None:
    sync = make_clocks(scenarios.load_fleet("2drones"))
    staggered = make_clocks(scenarios.load_fleet("2drones_staggered"))
    assert staggered["d1"].offset_s == CYCLE / 2 and staggered["d0"].offset_s == 0.0

    def both_charging_s(clocks: dict[str, BatteryClock]) -> int:
        return sum(all(c.charging(float(t)) for c in clocks.values()) for t in range(int(CYCLE)))

    assert both_charging_s(sync) == CHARGE  # the whole charging window, together
    assert both_charging_s(staggered) == 2 * CHARGE - CYCLE  # 600 s: what cannot be avoided


def test_agents_that_never_charge_have_no_clock() -> None:
    fleet = scenarios.load_fleet("2drones")
    walker = fleet.agents[0].model_copy(update={"id": "guard", "charge_time_s": 0.0})
    mixed = fleet.model_copy(update={"agents": [*fleet.agents, walker]})
    assert sorted(make_clocks(mixed)) == ["d0", "d1"]
    pos = np.array([10.0, 10.0])
    guard = AgentState("guard", 2, pos, 0.0, pos.copy(), 1.4, 20.0, 120.0, home=pos.copy())
    for t_abs in (0.0, 1600.0, 5000.0):
        assert step_battery([guard], make_clocks(mixed), t_abs, DT) == []
    assert guard.mode == PATROL and guard.active
