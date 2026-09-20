"""Things the sensors can look at: the intruder, benign traffic and the decoy.

Everything here reads the contracts only through adapt. Time is the episode clock: t = 0 is the
moment the intruder stands on its entry point.

Random streams: benign arrivals use one generator per route,
default_rng([seed, 1, crc32(route_id) % 1000000]), so adding or removing a route never changes
another route's objects, and nothing about the fleet can change the traffic. Stream 2 (intruder)
is reserved and unused in v0: the intruder follows its tactic deterministically.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Protocol

import numpy as np

from airtight.sim import adapt
from airtight.sim.geometry import Polyline

if TYPE_CHECKING:
    import numpy.typing as npt

    from airtight.contracts import Site, Tactic

    Array = npt.NDArray[np.float64]

BENIGN_STREAM = 1
INTRUDER_STREAM = 2  # reserved, unused in v0


class SimObject(Protocol):
    """kind is "intruder", "decoy", or the benign class string."""

    object_id: str
    kind: str

    def alive(self, t: float) -> bool: ...

    def position(self, t: float) -> Array:
        """(2,) metres, always a copy."""
        ...


class Intruder:
    def __init__(self, site: Site, tactic: Tactic) -> None:
        self.object_id = "intruder"
        self.kind = "intruder"
        self.path = adapt.intruder_path(site, tactic)
        self._polyline = Polyline(self.path)
        self.speed_mps = adapt.intruder_speed_mps(tactic)
        self.t_reach = adapt.t_reach(site, tactic)
        self.t_cdp = adapt.t_cdp(site, tactic)

    def alive(self, t: float) -> bool:
        return t >= 0.0

    def position(self, t: float) -> Array:
        """On the entry point at t <= 0, on the asset from t_reach onwards."""
        if t >= self.t_reach:
            asset: Array = self.path[-1].copy()
            return asset
        xy, _ = self._polyline.position(self.speed_mps, t)  # a fresh array
        return xy


class BenignObject:
    def __init__(
        self,
        object_id: str,
        cls: str,
        points: Array,
        speed_mps: float,
        t_start: float,
        polyline: Polyline | None = None,
    ) -> None:
        self.object_id = object_id
        self.kind = cls
        self.points = points
        self._polyline = polyline if polyline is not None else Polyline(points)
        self.speed_mps = speed_mps
        self.t_start = t_start
        self.t_end = t_start + self._polyline.total / speed_mps

    def alive(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

    def position(self, t: float) -> Array:
        xy, _ = self._polyline.position(self.speed_mps, t - self.t_start)  # a fresh array
        return xy


class Decoy:
    def __init__(self, position: Array, t_on: float, t_off: float) -> None:
        self.object_id = "decoy"
        self.kind = "decoy"
        self.t_on = t_on
        self.t_off = t_off
        self._position = position.copy()

    def alive(self, t: float) -> bool:
        return self.t_on <= t <= self.t_off

    def position(self, t: float) -> Array:
        return self._position.copy()


def make_decoy(tactic: Tactic) -> Decoy | None:
    spec = adapt.decoy_spec(tactic)
    if spec is None:
        return None
    return Decoy(spec.position, spec.t_on, spec.t_off)


def benign_rng(seed: int, route_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, BENIGN_STREAM, zlib.crc32(route_id.encode()) % 1000000])


def spawn_benign(site: Site, t0: float, t1: float, seed: int) -> list[BenignObject]:
    """Poisson arrivals on every benign route, sorted by (route_id, t_start).

    Each route's window is [t0 - route_duration, t1]. Starting the window early is what makes
    objects that are already partway along exist at t0. A route with rate 0 yields nothing and
    draws nothing.
    """
    if t1 < t0:
        raise ValueError(f"spawn window needs t0 <= t1, got {t0} and {t1}")
    objects: list[BenignObject] = []
    for route in sorted(adapt.benign_routes(site), key=lambda r: r.route_id):
        if route.arrivals_per_hour <= 0:
            continue
        rng = benign_rng(seed, route.route_id)
        path = Polyline(route.points)
        window_start = t0 - path.total / route.speed_mps
        n = int(rng.poisson(route.arrivals_per_hour / 3600.0 * (t1 - window_start)))
        starts = np.sort(rng.uniform(window_start, t1, size=n))
        objects.extend(
            BenignObject(
                f"{route.route_id}-{i}",
                route.cls,
                route.points,
                route.speed_mps,
                float(t),
                polyline=path,
            )
            for i, t in enumerate(starts)
        )
    return objects
