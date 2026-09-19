from __future__ import annotations

import numpy as np
import pytest

from airtight.sim import adapt, scenarios
from airtight.sim.constants import NEVER_SEEN
from airtight.sim.coverage import (
    SAMPLE_S,
    coverage_profile,
    duty_intervals,
    uncovered_intervals,
    uncovered_s_per_hour,
    uncovered_s_per_hour_exact,
)
from airtight.sim.episode import EpisodeParams, official_params, simulate

ENDURANCE, CHARGE, CYCLE = 1500.0, 2100.0, 3600.0  # the yard_night drone
# The longest flight home on yard_night: a pad in one corner, the far corner of the fence, 8 m/s.
MAX_TRANSIT_S = float(np.hypot(280.0 - 40.0, 180.0 - 40.0)) / 8.0 + 2.0


def _spans(name: str) -> list[tuple[float, float]]:
    return [(g.start_s, g.end_s) for g in uncovered_intervals(scenarios.load_fleet(name))]


def test_duty_intervals_come_straight_from_the_clocks() -> None:
    assert duty_intervals(scenarios.load_fleet("2drones")) == {
        "d0": [(0.0, ENDURANCE)],
        "d1": [(0.0, ENDURANCE)],
    }
    staggered = duty_intervals(scenarios.load_fleet("2drones_staggered"))
    assert staggered == {"d0": [(0.0, 1500.0)], "d1": [(1800.0, 3300.0)]}
    thirds = duty_intervals(scenarios.load_fleet("3drones_staggered"))
    assert thirds["d1"] == [(0.0, 300.0), (2400.0, 3600.0)]  # a duty stretch that wraps the cycle
    assert thirds["d2"] == [(1200.0, 2700.0)]
    assert all(sum(hi - lo for lo, hi in iv) == ENDURANCE for iv in thirds.values())


def test_uncovered_intervals_exact_numbers() -> None:
    assert _spans("2drones") == [(ENDURANCE, CYCLE)]  # one interval of exactly charge_time_s
    assert _spans("2drones")[0][1] - _spans("2drones")[0][0] == CHARGE
    assert _spans("2drones_staggered") == [(1500.0, 1800.0), (3300.0, 3600.0)]  # two of 300 s
    assert _spans("3drones_staggered") == []
    assert _spans("1drone") == [(ENDURANCE, CYCLE)]


def test_uncovered_intervals_as_phases_and_per_hour() -> None:
    first, second = uncovered_intervals(scenarios.load_fleet("2drones_staggered"))
    assert (first.start_phase, first.end_phase) == (1500.0 / 3600.0, 0.5)
    assert (second.start_phase, second.end_phase) == (3300.0 / 3600.0, 1.0)
    assert first.duration_s == second.duration_s == 300.0
    assert uncovered_s_per_hour_exact(scenarios.load_fleet("2drones")) == 2100.0
    assert uncovered_s_per_hour_exact(scenarios.load_fleet("2drones_staggered")) == 600.0
    assert uncovered_s_per_hour_exact(scenarios.load_fleet("3drones_staggered")) == 0.0


def test_an_agent_that_never_charges_leaves_nothing_uncovered() -> None:
    fleet = scenarios.load_fleet("2drones")
    guard = fleet.agents[0].model_copy(update={"id": "guard", "charge_time_s": 0.0})
    mixed = fleet.model_copy(update={"agents": [*fleet.agents, guard]})
    assert sorted(duty_intervals(mixed)) == ["d0", "d1"]  # only agents that charge are listed
    assert uncovered_intervals(mixed) == []


@pytest.fixture(scope="module")
def profiles() -> dict[str, object]:
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    names = ("2drones", "2drones_staggered", "3drones_staggered")
    return {n: coverage_profile(site, scenarios.load_fleet(n), curves, seed=3) for n in names}


def test_profile_shape_and_counts(profiles: dict) -> None:  # type: ignore[type-arg]
    sync = profiles["2drones"]
    assert sync.duration_s == CYCLE and len(sync.t_s) == int(CYCLE / SAMPLE_S)
    assert sync.t_s[0] == 0.0 and sync.t_s[-1] == CYCLE - SAMPLE_S
    assert np.all(sync.patrolling <= sync.on_duty) and sync.on_duty.max() == 2
    assert np.all(sync.on_duty[sync.t_s >= ENDURANCE] == 0)  # charging starts on the clock
    assert np.all(sync.on_duty[sync.t_s < ENDURANCE] == 2)
    assert np.all(sync.max_staleness_s >= sync.mean_staleness_s)
    # nobody is up in the charging window, so the yard only gets staler
    window = sync.max_staleness_s[sync.t_s >= ENDURANCE]
    assert np.all(np.diff(window) > 0) and window[-1] > CHARGE


def test_simulated_gap_agrees_with_the_arithmetic_to_within_the_flight_home(
    profiles: dict,  # type: ignore[type-arg]
) -> None:
    per_hour = 3600.0 / CYCLE
    for name, n_gaps in (("2drones", 1), ("2drones_staggered", 2), ("3drones_staggered", 0)):
        exact = uncovered_s_per_hour_exact(scenarios.load_fleet(name))
        simulated = uncovered_s_per_hour(profiles[name])
        assert exact <= simulated <= exact + n_gaps * MAX_TRANSIT_S * per_hour + SAMPLE_S
    sync, staggered, thirds = (uncovered_s_per_hour(profiles[n]) for n in profiles)
    assert thirds == 0.0 < staggered < sync


def test_profile_is_deterministic_and_everyone_is_up_with_the_battery_off() -> None:
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    fleet = scenarios.load_fleet("2drones")
    a, b = (
        coverage_profile(site, fleet, curves, seed=3),
        coverage_profile(site, fleet, curves, seed=3),
    )
    assert all(np.array_equal(x, y) for x, y in zip(a[:5], b[:5], strict=True))
    off = coverage_profile(site, fleet, curves, EpisodeParams(battery=False), seed=3)
    assert np.all(off.patrolling == 2) and uncovered_s_per_hour(off) == 0.0


@pytest.mark.parametrize("fleet_name", ["2drones", "2drones_staggered"])
def test_episode_inside_an_uncovered_interval_never_sees_the_intruder(fleet_name: str) -> None:
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    assert adapt.fixed_sensors(site) == []
    fleet = scenarios.load_fleet(fleet_name)
    params = official_params()
    for gap in uncovered_intervals(fleet):
        phase = (gap.start_phase + gap.end_phase) / 2.0  # picked from the arithmetic
        tactic = scenarios.load_tactic("jog").model_copy(update={"phase": phase})
        # the whole episode, jitter included, has to fit inside the gap for the claim to hold
        assert (
            adapt.t_reach(site, tactic) + params.tail_s + 2 * params.phase_jitter_s < gap.duration_s
        )
        for seed in range(1000, 1008):
            scores = simulate(site, fleet, tactic, curves, seed, params)
            assert scores.n_looks == 0 and scores.intruder_peak == NEVER_SEEN


def test_only_restricts_the_question_but_keeps_the_fleets_phase_axis() -> None:
    fleet = scenarios.load_fleet("2drones")
    guard = fleet.agents[0].model_copy(update={"id": "guard", "charge_time_s": 0.0})
    mixed = fleet.model_copy(update={"agents": [*fleet.agents, guard]})
    assert uncovered_intervals(mixed) == []  # the guard is always up
    (gap,) = uncovered_intervals(mixed, only=["d0", "d1"])  # but both drones are down together
    assert (gap.start_s, gap.end_s) == (ENDURANCE, CYCLE)
    assert uncovered_intervals(mixed, only=["d0", "guard"]) == []
    assert uncovered_intervals(mixed, only=[]) == []
    assert uncovered_intervals(fleet, only=["d0", "d1"]) == uncovered_intervals(fleet)
    with pytest.raises(ValueError, match="nobody"):
        uncovered_intervals(fleet, only=["nobody"])
