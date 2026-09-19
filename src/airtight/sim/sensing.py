"""Reduced-order sensing and the log-likelihood-ratio track score.

An observer looks at an object only if the observer is active, the object is alive, its range
is at most the observer's footprint radius, and it is inside the observer's wedge. Outside that
nothing is drawn and nothing is scored.

The world and the score use different numbers on purpose. Whether a look HITS is drawn from the
truth: pd_per_look for an intruder or decoy, the contract's true false-positive rate for a
benign class. How much a hit or miss MOVES the score always uses the intruder's pd at that
range and ASSUMED_PFA, for every object, because the system does not know what it is looking
at. Association is by truth: every look is credited to the right object.

Clutter that is not tied to an object is ignored in v0.

Each observer's draws come from its own stream, default_rng([seed, 1000 + crc32(agent_id) %
1000000]), one rng.random() per qualifying observer-object pair per look.
"""

from __future__ import annotations

import math
import zlib
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from airtight.sim import adapt
from airtight.sim.constants import ASSUMED_PFA, NEVER_SEEN, SCORE_FLOOR, TAU_REF
from airtight.sim.geometry import in_wedge

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from airtight.contracts import SensorCurves
    from airtight.sim.actors import SimObject

    Array = npt.NDArray[np.float64]

SENSOR_STREAM_BASE = 1000
TARGET_KINDS = frozenset({"intruder", "decoy"})
_PD_CLIP = (0.001, 0.999)
_TIME_EPS = 1e-9


class Observer(Protocol):
    """Anything that looks: a patrolling agent now, a fixed sensor from step 2.5."""

    agent_id: str
    pos: Array
    heading: float
    fov_deg: float
    footprint_radius_m: float
    sensor_type: str
    active: bool


class Look(NamedTuple):
    agent_id: str
    object_id: str
    hit: bool
    score: float  # the object's score after this look


def sensor_rng(seed: int, agent_id: str) -> np.random.Generator:
    return np.random.default_rng(
        [seed, SENSOR_STREAM_BASE + zlib.crc32(agent_id.encode()) % 1000000]
    )


def look_range_m(observer: Observer, obj: SimObject, t: float) -> float | None:
    """The range if the observer looks at the object at t, else None. The one qualifying rule."""
    if not observer.active or not obj.alive(t):
        return None
    xy = obj.position(t)
    range_m = float(np.hypot(xy[0] - observer.pos[0], xy[1] - observer.pos[1]))
    if range_m > observer.footprint_radius_m:
        return None
    if not bool(in_wedge(observer.pos, observer.heading, observer.fov_deg, xy)):
        return None
    return range_m


def hit_probability(
    sensor_curves: SensorCurves, sensor_type: str, kind: str, range_m: float
) -> float:
    """The TRUE chance this look reports a detection."""
    if kind in TARGET_KINDS:
        return adapt.pd_per_look(sensor_curves, sensor_type, range_m)
    return adapt.true_fp_per_look(sensor_curves, sensor_type, kind)


def llr_increment(pd: float, hit: bool) -> float:
    """Score change for one look. pd is the INTRUDER pd at that range, whatever the object is."""
    p = min(max(pd, _PD_CLIP[0]), _PD_CLIP[1])
    if hit:
        return math.log(p / ASSUMED_PFA)
    return math.log((1.0 - p) / (1.0 - ASSUMED_PFA))


class ScoreBook:
    """Per-object track scores, with a compact running-peak series for offline thresholds."""

    def __init__(self) -> None:
        self._score: dict[str, float] = {}
        self._peaks: dict[str, list[tuple[float, float]]] = {}

    def update(self, object_id: str, increment: float, t: float) -> float:
        """Add to the score, floored at SCORE_FLOOR. Calls must come in non-decreasing t."""
        score = max(SCORE_FLOOR, self._score.get(object_id, 0.0) + increment)
        self._score[object_id] = score
        series = self._peaks.setdefault(object_id, [])
        if not series or score > series[-1][1]:
            series.append((t, score))
        return score

    def score(self, object_id: str) -> float | None:
        return self._score.get(object_id)

    def peak(self, object_id: str, t_max: float | None = None) -> float:
        """Highest score at or before t_max; NEVER_SEEN if never looked at by then."""
        best = NEVER_SEEN
        for t, value in self._peaks.get(object_id, []):
            if t_max is not None and t > t_max + _TIME_EPS:
                break
            best = value
        return best

    def first_crossing(self, object_id: str, tau: float = TAU_REF) -> float | None:
        """Earliest t at which the score was >= tau, else None.

        The running peak rises exactly when the score sets a new maximum, so the first peak at
        or above tau is the first time the score got there.
        """
        for t, value in self._peaks.get(object_id, []):
            if value >= tau:
                return t
        return None

    def peak_series(self, object_id: str) -> list[tuple[float, float]]:
        return list(self._peaks.get(object_id, []))

    def object_ids(self) -> list[str]:
        return sorted(self._score)


class LookSchedule:
    """Per-observer look times: t = 0, then every 1 / rate seconds. Never due for t < 0.

    The sim steps in dt, so a look is due at the first step at or after its look time. For a
    period that is a whole number of steps that is exactly the look time.
    """

    def __init__(self, look_rate_hz_by_agent_id: Mapping[str, float], dt: float) -> None:
        self.dt = dt
        self.period_s: dict[str, float] = {}
        for agent_id, rate_hz in look_rate_hz_by_agent_id.items():
            period = 1.0 / rate_hz
            if period < dt - _TIME_EPS:
                raise ValueError(
                    f"{agent_id!r} looks every {period:.4f} s, shorter than the sim step {dt} s"
                )
            self.period_s[agent_id] = period

    def due(self, agent_id: str, t: float) -> bool:
        if t < -_TIME_EPS:
            return False
        period = self.period_s[agent_id]
        looks_by_now = math.floor((t + _TIME_EPS) / period)
        looks_by_last_step = math.floor((t - self.dt + _TIME_EPS) / period)
        return looks_by_now > looks_by_last_step


def do_looks(
    observers: Sequence[Observer],
    objects: Sequence[SimObject],
    t: float,
    sensor_curves: SensorCurves,
    rngs: Mapping[str, np.random.Generator],
    schedule: LookSchedule,
    book: ScoreBook,
) -> list[Look]:
    """One sensing pass at time t. Returns every look made, for logging.

    Observers go in agent_id order and objects in object_id order. Draw order only matters per
    observer, but update order matters across observers because the score floor does not
    commute: a hit then a miss at the floor differs from a miss then a hit.
    """
    looks: list[Look] = []
    for observer in sorted(observers, key=lambda o: o.agent_id):
        if not observer.active or not schedule.due(observer.agent_id, t):
            continue
        rng = rngs[observer.agent_id]
        for obj in sorted(objects, key=lambda o: o.object_id):
            range_m = look_range_m(observer, obj, t)
            if range_m is None:
                continue
            p_hit = hit_probability(sensor_curves, observer.sensor_type, obj.kind, range_m)
            hit = bool(rng.random() < p_hit)
            pd = adapt.pd_per_look(sensor_curves, observer.sensor_type, range_m)
            score = book.update(obj.object_id, llr_increment(pd, hit), t)
            looks.append(Look(observer.agent_id, obj.object_id, hit, score))
    return looks
