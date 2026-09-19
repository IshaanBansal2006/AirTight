from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from airtight.contracts import SensorCurve, SensorCurves
from airtight.sim import adapt
from airtight.sim.constants import ASSUMED_PFA, NEVER_SEEN, SCORE_FLOOR, TAU_REF
from airtight.sim.fleet import AgentState
from airtight.sim.sensing import (
    LookSchedule,
    ScoreBook,
    do_looks,
    hit_probability,
    llr_increment,
    look_range_m,
    sensor_rng,
)

DT = 0.25


def _curve(sensor_type: str, look_rate_hz: float, fov_deg: float = 360.0) -> SensorCurve:
    return SensorCurve(
        sensor_type=sensor_type,
        range_bins_m=[5, 10, 15, 20, 30],
        pd_per_look=[0.7, 0.6, 0.5, 0.4, 0.0],
        pfa_per_look_by_class={"animal": 0.08, "debris": 0.03},
        fov_deg=fov_deg,
        look_rate_hz=look_rate_hz,
    )


CURVES = SensorCurves(
    source="synthetic test curve",
    curves={"cam": _curve("cam", 2.0), "slow_cam": _curve("slow_cam", 0.5)},
)


class Held:
    """An object that stands still."""

    def __init__(self, object_id: str, kind: str, xy: tuple[float, float], alive: bool = True):
        self.object_id = object_id
        self.kind = kind
        self._xy = np.array(xy, dtype=np.float64)
        self._alive = alive

    def alive(self, t: float) -> bool:
        return self._alive

    def position(self, t: float) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._xy.copy()


def _observer(
    agent_id: str = "d0",
    xy: tuple[float, float] = (100.0, 100.0),
    fov: float = 360.0,
    heading: float = 0.0,
    sensor_type: str = "cam",
) -> AgentState:
    pos = np.array(xy, dtype=np.float64)
    radius = adapt.sensor_footprint_radius_m(CURVES, sensor_type)
    return AgentState(
        agent_id, 0, pos, heading, pos.copy(), 8.0, radius, fov, sensor_type=sensor_type
    )


def _schedule(*observers: AgentState) -> LookSchedule:
    rates = {o.agent_id: adapt.sensor_look_rate_hz(CURVES, o.sensor_type) for o in observers}
    return LookSchedule(rates, DT)


def _state(rng: np.random.Generator) -> dict[str, Any]:
    return dict(rng.bit_generator.state)


def _hold(objects: list[Held], seed: int, n_looks: int = 20) -> ScoreBook:
    """One 2 Hz observer at (100, 100) staring at the objects for n_looks looks."""
    observer = _observer()
    book, rngs, schedule = ScoreBook(), {"d0": sensor_rng(seed, "d0")}, _schedule(observer)
    for k in range(n_looks * 2):
        do_looks([observer], objects, k * DT, CURVES, rngs, schedule, book)
    return book


def test_footprint_of_the_synthetic_curve_is_the_last_live_bin() -> None:
    assert _observer().footprint_radius_m == 20.0


def test_llr_increment_signs_values_and_no_inf_or_nan() -> None:
    assert llr_increment(0.6, True) == pytest.approx(math.log(0.6 / ASSUMED_PFA))
    assert llr_increment(0.6, False) == pytest.approx(math.log(0.4 / (1 - ASSUMED_PFA)))
    assert llr_increment(0.6, True) > 0 > llr_increment(0.6, False)
    for pd in (0.0, 1.0):
        for hit in (True, False):
            assert math.isfinite(llr_increment(pd, hit))


def test_hit_probability_uses_truth_per_kind() -> None:
    assert hit_probability(CURVES, "cam", "intruder", 8.0) == 0.6
    assert hit_probability(CURVES, "cam", "decoy", 8.0) == 0.6
    assert hit_probability(CURVES, "cam", "intruder", 5.0) == 0.7  # an edge closes its own bin
    assert hit_probability(CURVES, "cam", "intruder", 31.0) == 0.0
    assert hit_probability(CURVES, "cam", "animal", 8.0) == 0.08
    assert hit_probability(CURVES, "cam", "debris", 19.0) == 0.03  # one scalar, any range
    with pytest.raises(KeyError, match="unicorn"):
        hit_probability(CURVES, "cam", "unicorn", 8.0)


def test_intruder_is_caught_and_debris_is_not() -> None:
    caught = sum(
        _hold([Held("intruder", "intruder", (108.0, 100.0))], seed).peak("intruder") >= TAU_REF
        for seed in range(100)
    )
    false_alarms = sum(
        _hold([Held("tarp-0", "debris", (108.0, 100.0))], seed).peak("tarp-0") >= TAU_REF
        for seed in range(100)
    )
    assert caught >= 90
    assert false_alarms <= 10


def test_a_benign_hit_moves_the_score_exactly_like_an_intruder_hit() -> None:
    # Same range, so the same increments: the system does not know what it is looking at.
    book = _hold([Held("fox-0", "animal", (108.0, 100.0))], seed=3, n_looks=1)
    assert book.score("fox-0") in (
        pytest.approx(llr_increment(0.6, True)),
        pytest.approx(llr_increment(0.6, False)),
    )


@pytest.mark.parametrize(
    ("observer", "xy"),
    [
        (_observer(), (120.5, 100.0)),  # just beyond the 20 m footprint
        (_observer(fov=90.0), (92.0, 100.0)),  # 8 m behind a 90 degree wedge looking along +x
        (_observer(fov=90.0), (100.0, 108.0)),  # 8 m to the side of it
    ],
)
def test_unqualified_object_is_never_seen_and_nothing_is_drawn(
    observer: AgentState, xy: tuple[float, float]
) -> None:
    target = Held("intruder", "intruder", xy)
    assert look_range_m(observer, target, 0.0) is None
    book, rngs = ScoreBook(), {"d0": sensor_rng(5, "d0")}
    looks = do_looks([observer], [target], 0.0, CURVES, rngs, _schedule(observer), book)
    assert looks == [] and book.object_ids() == []
    assert book.peak("intruder") == NEVER_SEEN
    assert _state(rngs["d0"]) == _state(sensor_rng(5, "d0"))


def test_wedge_sees_ahead_and_a_dead_object_is_skipped() -> None:
    observer = _observer(fov=90.0)
    assert look_range_m(observer, Held("intruder", "intruder", (108.0, 100.0)), 0.0) == 8.0
    assert look_range_m(observer, Held("gone", "animal", (108.0, 100.0), alive=False), 0.0) is None
    edge = Held("intruder", "intruder", (120.0, 100.0))  # exactly on the footprint radius
    assert look_range_m(observer, edge, 0.0) == 20.0


def test_inactive_observer_never_looks_and_never_draws() -> None:
    observer = _observer()
    observer.active = False
    target = Held("intruder", "intruder", (108.0, 100.0))
    book, rngs = ScoreBook(), {"d0": sensor_rng(5, "d0")}
    for k in range(40):
        assert do_looks([observer], [target], k * DT, CURVES, rngs, _schedule(observer), book) == []
    assert book.peak("intruder") == NEVER_SEEN
    assert _state(rngs["d0"]) == _state(sensor_rng(5, "d0"))


def test_exactly_one_draw_per_qualifying_pair() -> None:
    observer = _observer()
    objects = [
        Held("intruder", "intruder", (108.0, 100.0)),
        Held("fox-0", "animal", (100.0, 105.0)),
        Held("far-0", "animal", (300.0, 300.0)),
    ]
    book, rngs = ScoreBook(), {"d0": sensor_rng(5, "d0")}
    looks = do_looks([observer], objects, 0.0, CURVES, rngs, _schedule(observer), book)
    assert [look.object_id for look in looks] == ["fox-0", "intruder"]
    reference = sensor_rng(5, "d0")
    reference.random(2)
    assert _state(rngs["d0"]) == _state(reference)


def test_score_book_floor_peak_and_t_max() -> None:
    book = ScoreBook()
    assert book.peak("x") == NEVER_SEEN and book.first_crossing("x") is None
    book.update("x", -1.0, 0.0)
    assert book.peak("x") == -1.0  # looked at and missed is not NEVER_SEEN
    for k in range(1, 20):
        book.update("x", -1.0, float(k))
    assert book.score("x") == SCORE_FLOOR
    book.update("x", 7.0, 20.0)
    book.update("x", 3.0, 21.0)
    assert book.score("x") == 5.0 and book.peak("x") == 5.0
    assert book.peak("x", t_max=19.5) == -1.0  # ignores the later hits
    assert book.peak("x", t_max=20.0) == 2.0
    assert book.peak("x", t_max=-1.0) == NEVER_SEEN
    assert book.object_ids() == ["x"]


def test_first_crossing_is_the_earliest_time_and_does_not_move() -> None:
    book = ScoreBook()
    book.update("x", 3.0, 1.0)
    assert book.first_crossing("x") is None
    book.update("x", 1.5, 2.0)
    assert book.first_crossing("x") == 2.0
    book.update("x", -4.0, 3.0)  # drops back below
    book.update("x", 6.0, 4.0)  # and crosses again, higher
    assert book.first_crossing("x") == 2.0
    assert book.first_crossing("x", tau=6.0) == 4.0
    assert book.first_crossing("x", tau=100.0) is None
    assert book.peak_series("x") == [(1.0, 3.0), (2.0, 4.5), (4.0, 6.5)]  # only rises are kept


def test_look_schedule_counts_and_validation() -> None:
    schedule = LookSchedule({"fast": 2.0, "slow": 0.5}, DT)
    times = [k * DT for k in range(-8, 41)]  # -2 s to 10 s
    assert sum(schedule.due("fast", t) for t in times) == 21
    assert sum(schedule.due("slow", t) for t in times) == 6
    assert [t for t in times if schedule.due("slow", t)] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert not any(schedule.due(a, t) for a in ("fast", "slow") for t in times if t < 0)
    with pytest.raises(ValueError, match="shorter than the sim step"):
        LookSchedule({"too_fast": 10.0}, DT)
    LookSchedule({"every_step": 4.0}, DT)  # a period equal to dt is fine


def test_slow_sensor_looks_a_quarter_as_often() -> None:
    fast, slow = _observer("fast"), _observer("slow", sensor_type="slow_cam")
    target = Held("intruder", "intruder", (108.0, 100.0))
    rngs = {"fast": sensor_rng(1, "fast"), "slow": sensor_rng(1, "slow")}
    book, schedule = ScoreBook(), _schedule(fast, slow)
    looks = [
        look
        for k in range(41)
        for look in do_looks([fast, slow], [target], k * DT, CURVES, rngs, schedule, book)
    ]
    assert sum(look.agent_id == "fast" for look in looks) == 21
    assert sum(look.agent_id == "slow" for look in looks) == 6


def _run_pair(seed: int, observers_reversed: bool, objects_reversed: bool) -> ScoreBook:
    observers = [_observer("d0", (100.0, 100.0)), _observer("d1", (110.0, 100.0))]
    objects = [
        Held("intruder", "intruder", (106.0, 100.0)),
        Held("fox-0", "animal", (104.0, 103.0)),
        Held("tarp-0", "debris", (103.0, 97.0)),
    ]
    rngs = {o.agent_id: sensor_rng(seed, o.agent_id) for o in observers}
    book, schedule = ScoreBook(), _schedule(*observers)
    if observers_reversed:
        observers.reverse()
    if objects_reversed:
        objects.reverse()
    for k in range(60):
        do_looks(observers, objects, k * DT, CURVES, rngs, schedule, book)
    return book


def _dump(book: ScoreBook) -> dict[str, list[tuple[float, float]]]:
    return {oid: book.peak_series(oid) for oid in book.object_ids()}


def test_same_seed_same_result_and_order_of_inputs_does_not_matter() -> None:
    base = _dump(_run_pair(9, False, False))
    assert set(base) == {"intruder", "fox-0", "tarp-0"}
    assert _dump(_run_pair(9, False, False)) == base
    assert _dump(_run_pair(9, False, True)) == base  # objects reordered
    assert _dump(_run_pair(9, True, False)) == base  # observers reordered
    assert _dump(_run_pair(10, False, False)) != base


def test_sensor_rng_depends_on_seed_and_agent_only() -> None:
    assert _state(sensor_rng(1, "d0")) == _state(sensor_rng(1, "d0"))
    assert _state(sensor_rng(1, "d0")) != _state(sensor_rng(1, "d1"))
    assert _state(sensor_rng(1, "d0")) != _state(sensor_rng(2, "d0"))
