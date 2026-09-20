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

Draws come from one stream per observer and object pair,
default_rng([seed, 5, crc32(observer_id), crc32(object_id)]), one rng.random() per qualifying
look. An object's hit sequence therefore depends only on the geometry between it and that
observer: another object wandering into view cannot shift it, which is what paired comparisons
need.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from airtight.sim import adapt
from airtight.sim.constants import ASSUMED_PFA, NEVER_SEEN, SCORE_FLOOR, TAU_REF, TIME_EPS
from airtight.sim.geometry import in_wedge

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from airtight.contracts import SensorCurves, Site
    from airtight.sim.actors import SimObject

    Array = npt.NDArray[np.float64]

LOOK_STREAM = 5
TARGET_KINDS = frozenset({"intruder", "decoy"})
_PD_CLIP = (0.001, 0.999)


class Observer(Protocol):
    """Anything that looks: a patrolling agent now, a fixed sensor from step 2.5."""

    agent_id: str
    pos: Array
    heading: float
    fov_deg: float
    footprint_radius_m: float
    sensor_type: str
    active: bool


@dataclass
class FixedObserver:
    """A fixed sensor: an observer that never moves and is always active."""

    agent_id: str
    pos: Array
    heading: float
    fov_deg: float
    footprint_radius_m: float
    sensor_type: str
    active: bool = True


def make_fixed_observers(site: Site, sensor_curves: SensorCurves) -> list[FixedObserver]:
    return [
        FixedObserver(
            agent_id=spec.sensor_id,
            pos=spec.position,
            heading=spec.heading_rad,
            fov_deg=adapt.sensor_fov_deg(sensor_curves, spec.sensor_type),
            footprint_radius_m=adapt.sensor_footprint_radius_m(sensor_curves, spec.sensor_type),
            sensor_type=spec.sensor_type,
        )
        for spec in adapt.fixed_sensors(site)
    ]


class Look(NamedTuple):
    agent_id: str
    object_id: str
    hit: bool
    score: float  # the object's score after this look


def look_rng(seed: int, observer_id: str, object_id: str) -> np.random.Generator:
    return np.random.default_rng(
        [seed, LOOK_STREAM, zlib.crc32(observer_id.encode()), zlib.crc32(object_id.encode())]
    )


class LookRngs:
    """The episode's look generators, one per observer and object pair, created on first use."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rngs: dict[tuple[str, str], np.random.Generator] = {}

    def get(self, observer_id: str, object_id: str) -> np.random.Generator:
        key = (observer_id, object_id)
        if key not in self._rngs:
            self._rngs[key] = look_rng(self.seed, observer_id, object_id)
        return self._rngs[key]

    def pairs(self) -> list[tuple[str, str]]:
        """Every pair that has been looked at so far, sorted."""
        return sorted(self._rngs)


def look_range_m(
    observer: Observer, obj: SimObject, t: float, xy: Array | None = None
) -> float | None:
    """The range if the observer looks at the object at t, else None. The one qualifying rule.

    Pass xy when the caller already has obj.position(t); the function otherwise asks again.
    """
    if not observer.active or not obj.alive(t):
        return None
    if xy is None:
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
            if t_max is not None and t > t_max + TIME_EPS:
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
            if period < dt - TIME_EPS:
                raise ValueError(
                    f"{agent_id!r} looks every {period:.4f} s, shorter than the sim step {dt} s"
                )
            self.period_s[agent_id] = period

    def due(self, agent_id: str, t: float) -> bool:
        if t < -TIME_EPS:
            return False
        period = self.period_s[agent_id]
        looks_by_now = math.floor((t + TIME_EPS) / period)
        looks_by_last_step = math.floor((t - self.dt + TIME_EPS) / period)
        return looks_by_now > looks_by_last_step


def do_looks(
    observers: Sequence[Observer],
    objects: Sequence[SimObject],
    t: float,
    sensor_curves: SensorCurves,
    rngs: LookRngs,
    schedule: LookSchedule,
    book: ScoreBook,
) -> list[Look]:
    """One sensing pass at time t. Returns every look made, for logging.

    Observers go in agent_id order and objects in object_id order. With one stream per pair the
    draws no longer depend on order, but the score updates still do, because the score floor
    does not commute: a hit then a miss at the floor differs from a miss then a hit.
    """
    looks: list[Look] = []
    due = [
        observer
        for observer in sorted(observers, key=lambda o: o.agent_id)
        if observer.active and schedule.due(observer.agent_id, t)
    ]
    if not due:
        return looks
    alive = sorted((obj for obj in objects if obj.alive(t)), key=lambda o: o.object_id)
    positions = [obj.position(t) for obj in alive]
    for observer in due:
        for obj, xy in zip(alive, positions, strict=True):
            range_m = look_range_m(observer, obj, t, xy)
            if range_m is None:
                continue
            if obj.kind in TARGET_KINDS:
                pd = p_hit = adapt.pd_per_look(sensor_curves, observer.sensor_type, range_m)
            else:
                p_hit = hit_probability(sensor_curves, observer.sensor_type, obj.kind, range_m)
                pd = adapt.pd_per_look(sensor_curves, observer.sensor_type, range_m)
            hit = bool(rngs.get(observer.agent_id, obj.object_id).random() < p_hit)
            score = book.update(obj.object_id, llr_increment(pd, hit), t)
            looks.append(Look(observer.agent_id, obj.object_id, hit, score))
    return looks
