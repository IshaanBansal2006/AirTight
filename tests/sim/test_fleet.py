from __future__ import annotations

import math
from importlib import resources
from typing import Any

import numpy as np
import pytest

from airtight.contracts import FleetConfig, SensorCurves, Site
from airtight.sim import adapt
from airtight.sim import fleet as fleet_module
from airtight.sim.fleet import (
    AgentState,
    PatrolController,
    make_agents,
    patrol_rng,
    step_agents,
)
from airtight.sim.geometry import Grid, inside_mask, patrol_weight

# A synthetic yard built from geometry only; the contract example is too small for patrol tests.
GRID = Grid(0.0, 0.0, 300.0, 200.0, 5.0)
FENCE = np.array([[20.0, 20.0], [280.0, 20.0], [280.0, 180.0], [20.0, 180.0]])
INSIDE = inside_mask(GRID, FENCE)
WEIGHT = patrol_weight(GRID, INSIDE, np.array([[150.0, 100.0]]))
DT = 0.25
STEPS_PER_RETARGET = 4  # the episode loop retargets once a second

EXAMPLES = resources.files("airtight.contracts.examples")


def _agent(
    agent_id: str,
    index: int,
    xy: tuple[float, float],
    speed: float = 8.0,
    radius: float = 20.0,
    fov: float = 360.0,
    heading: float = 0.0,
) -> AgentState:
    pos = np.array(xy, dtype=np.float64)
    return AgentState(agent_id, index, pos, heading, pos.copy(), speed, radius, fov)


def _two_drones() -> list[AgentState]:
    return [_agent("d0", 0, (40.0, 40.0)), _agent("d1", 1, (260.0, 160.0))]


def _run(
    agents: list[AgentState], ctrl: PatrolController, t0: float, seconds: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """The patrol part of the episode loop. Returns positions, shape (steps, agents, 2)."""
    trail = []
    for k in range(round(seconds / DT)):
        t = t0 + k * DT
        ctrl.mark_seen(agents, t)
        if k % STEPS_PER_RETARGET == 0:
            ctrl.retarget(agents, t)
        step_agents(agents, DT)
        trail.append(np.array([a.pos for a in agents]))
    return np.array(trail)


def _state(rng: np.random.Generator) -> dict[str, Any]:
    return dict(rng.bit_generator.state)


def test_same_seed_same_trajectories_and_different_seed_differs() -> None:
    runs = []
    for seed in (11, 11, 12):
        agents = _two_drones()
        runs.append(_run(agents, PatrolController(GRID, WEIGHT, seed, t_start=0.0), 0.0, 60.0))
    assert np.array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


def test_agent_generator_does_not_depend_on_fleet_size_or_creation_order() -> None:
    fresh = _state(patrol_rng(5, "d0"))
    ctrl = PatrolController(GRID, WEIGHT, 5, t_start=0.0)
    ctrl.rng_for("d2")
    ctrl.rng_for("d1")
    assert _state(ctrl.rng_for("d0")) == fresh
    assert _state(patrol_rng(5, "d1")) != fresh
    assert _state(patrol_rng(6, "d0")) != fresh


@pytest.mark.parametrize("n_others", [1, 2])
def test_other_agents_draws_never_consume_agent_zeros_stream(n_others: int) -> None:
    ctrl = PatrolController(GRID, WEIGHT, 5, t_start=0.0)
    d0 = _agent("d0", 0, (40.0, 40.0))
    d0.target = np.array([260.0, 40.0])  # far away, and it has just retargeted: no draw due
    d0.last_retarget_t = 0.0
    others = [_agent(f"d{i}", i, (100.0 + 60.0 * i, 150.0)) for i in range(1, n_others + 1)]
    ctrl.retarget([d0, *others], 1.0)
    assert all(o.last_retarget_t == 1.0 for o in others)  # they did draw
    assert _state(ctrl.rng_for("d0")) == _state(patrol_rng(5, "d0"))
    assert d0.target.tolist() == [260.0, 40.0]


def test_agent_arrives_in_exactly_twenty_steps_without_overshoot() -> None:
    agent = _agent("d0", 0, (50.0, 50.0))
    agent.target = np.array([74.0, 82.0])  # 40 m away, off-axis so float error is exercised
    remaining = [40.0]
    steps = 0
    while not np.array_equal(agent.pos, agent.target):
        step_agents([agent], DT)
        steps += 1
        remaining.append(float(np.linalg.norm(agent.target - agent.pos)))
        assert remaining[-1] < remaining[-2]  # always closer, never past
        assert steps <= 25
    assert steps == 20
    assert math.isclose(agent.heading, math.atan2(32.0, 24.0))
    step_agents([agent], DT)
    assert agent.pos.tolist() == [74.0, 82.0]


def test_two_drones_see_most_of_the_yard_in_two_minutes() -> None:
    agents = _two_drones()
    ctrl = PatrolController(GRID, WEIGHT, 1000, t_start=0.0)
    _run(agents, ctrl, 0.0, 120.0)
    seen_fraction = float((ctrl.last_seen[INSIDE] >= 0.0).mean())
    assert seen_fraction > 0.60


def test_it_never_parks() -> None:
    agents = _two_drones()
    ctrl = PatrolController(GRID, WEIGHT, 1000, t_start=-60.0)
    _run(agents, ctrl, -60.0, 60.0)  # warm-up
    trail = _run(agents, ctrl, 0.0, 120.0)
    travelled = np.linalg.norm(np.diff(trail, axis=0), axis=2).sum(axis=0)
    assert np.all(travelled > 0.5 * 8.0 * 120.0)


def test_staleness_is_never_negative_and_zero_where_weight_is_zero() -> None:
    ctrl = PatrolController(GRID, WEIGHT, 1, t_start=-150.0)
    assert np.all(ctrl.staleness(-150.0)[INSIDE] == 300.0)
    ctrl.mark_seen(_two_drones(), 0.0)
    stale = ctrl.staleness(-10.0)  # asking about a time before a cell was seen
    assert np.all(stale >= 0.0) and np.all(stale[~INSIDE] == 0.0)


def test_inactive_agent_is_ignored_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_peers: list[set[int]] = []
    real = fleet_module.voronoi_mask

    def spy(centres: Any, own_xy: Any, peers: dict[int, Any], own_id: int) -> Any:
        seen_peers.append(set(peers))
        return real(centres, own_xy, peers, own_id)

    monkeypatch.setattr(fleet_module, "voronoi_mask", spy)
    active, idle = _two_drones()
    idle.active = False
    idle.target = np.array([100.0, 100.0])
    ctrl = PatrolController(GRID, WEIGHT, 3, t_start=0.0)

    before = ctrl.last_seen.copy()
    ctrl.mark_seen([idle], 5.0)
    assert np.array_equal(ctrl.last_seen, before)  # marks nothing seen

    ctrl.retarget([active, idle], 5.0)
    assert seen_peers == [set()]  # left out of the active agent's Voronoi region
    assert _state(ctrl.rng_for("d1")) == _state(patrol_rng(3, "d1"))  # drew nothing
    assert idle.target.tolist() == [100.0, 100.0] and idle.last_retarget_t == -math.inf

    step_agents([active, idle], DT)
    assert idle.pos.tolist() == [260.0, 160.0]  # not moved


def test_field_of_view_wedge_sees_ahead_not_behind() -> None:
    agent = _agent("g0", 0, (152.5, 102.5), fov=90.0, heading=0.0)  # looking along +x
    ctrl = PatrolController(GRID, WEIGHT, 1, t_start=0.0)
    ctrl.mark_seen([agent], 7.0)
    assert ctrl.last_seen[GRID.cell_of(162.5, 102.5)] == 7.0  # 10 m ahead
    assert ctrl.last_seen[GRID.cell_of(142.5, 102.5)] == -300.0  # 10 m behind
    assert ctrl.last_seen[GRID.cell_of(152.5, 112.5)] == -300.0  # 10 m to the side
    assert ctrl.last_seen[GRID.cell_of(152.5, 102.5)] == 7.0  # its own cell


def test_nothing_useful_keeps_the_old_target_and_draws_nothing() -> None:
    agent = _agent("d0", 0, (40.0, 40.0))
    ctrl = PatrolController(GRID, np.zeros(GRID.shape), 9, t_start=0.0)
    ctrl.retarget([agent], 1.0)
    assert agent.target.tolist() == [40.0, 40.0] and agent.last_retarget_t == -math.inf
    assert _state(ctrl.rng_for("d0")) == _state(patrol_rng(9, "d0"))


def test_controller_rejects_a_weight_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        PatrolController(GRID, np.zeros((3, 3)), 1, t_start=0.0)


def test_make_agents_on_the_contract_example() -> None:
    site = Site.model_validate_json(EXAMPLES.joinpath("site.json").read_text())
    fleet = FleetConfig.model_validate_json(EXAMPLES.joinpath("fleet_config.json").read_text())
    curves = SensorCurves.model_validate_json(EXAMPLES.joinpath("sensor_curve.json").read_text())
    agents = make_agents(site, fleet, curves)
    assert [a.agent_id for a in agents] == adapt.agent_ids(fleet)
    assert [a.index for a in agents] == [0, 1, 2, 3]
    for a in agents:
        sensor = adapt.agent_sensor_type(fleet, a.agent_id)
        assert a.speed_mps == adapt.agent_speed_mps(fleet, a.agent_id)
        assert a.footprint_radius_m == adapt.sensor_footprint_radius_m(curves, sensor)
        assert a.fov_deg == adapt.sensor_fov_deg(curves, sensor)
        assert np.array_equal(a.pos, a.target) and a.pos is not a.target
        assert a.active and a.last_retarget_t == -math.inf
    for i in range(4):
        for j in range(i + 1, 4):
            assert float(np.linalg.norm(agents[i].pos - agents[j].pos)) > 1.0
