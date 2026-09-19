"""The only module in sim/ that reads ambiguous contract fields.

Contract models go in; plain floats and numpy arrays come out, so a contract change lands here
and nowhere else in the lane.

Time: t = 0 is the moment the intruder stands on its entry point, which is what the log
contract ("seconds since the intruder entered") and the stub runner both mean. Warm-up runs at
negative t on the same clock and is never logged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from airtight.contracts import (
        AgentSpec,
        FleetConfig,
        SensorCurve,
        SensorCurves,
        Site,
        Tactic,
    )

    Array = npt.NDArray[np.float64]

START_OFFSET_M = 2.0
BENIGN_SPEED_MPS = {"person": 1.4, "vehicle": 5.0, "animal": 2.0, "debris": 0.5}
DEFAULT_BENIGN_SPEED_MPS = 1.4


def assets(site: Site) -> Array:
    """Asset positions, shape (n, 2). The contract has one asset today."""
    return np.array([[site.asset.x, site.asset.y]], dtype=np.float64)


def intruder_path(site: Site, tactic: Tactic) -> Array:
    """Entry point then the tactic's waypoints, shape (n, 2). Same path as the stub runner."""
    pts = [site.entry(tactic.entry_id).position, *tactic.waypoints]
    return np.array([[p.x, p.y] for p in pts], dtype=np.float64)


def path_length_m(path: Array) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def t_reach(site: Site, tactic: Tactic) -> float:
    """Seconds after entry at which the intruder reaches the asset."""
    return path_length_m(intruder_path(site, tactic)) / tactic.speed_mps


def t_cdp(site: Site, tactic: Tactic) -> float:
    """Critical detection point: t_reach minus the response time, clamped at 0 as in the stub."""
    return max(0.0, t_reach(site, tactic) - site.response_time_s)


def start_position(site: Site, fleet: FleetConfig, agent: AgentSpec) -> Array:
    """Docks round-robin by fleet index, else the perimeter centroid, plus a small offset.

    The offset is START_OFFSET_M at angle 2*pi*index/n. Two agents on the exact same point tie
    everywhere in a Voronoi test and the higher index never gets a region. Dock capacity is
    ignored here.
    """
    ids = [a.id for a in fleet.agents]
    if agent.id not in ids:
        raise ValueError(f"agent {agent.id!r} is not in fleet {fleet.name!r}; known: {ids}")
    index, n = ids.index(agent.id), len(ids)
    if site.docks:
        dock = site.docks[index % len(site.docks)].position
        base = np.array([dock.x, dock.y], dtype=np.float64)
    else:
        base = np.array([[p.x, p.y] for p in site.perimeter], dtype=np.float64).mean(axis=0)
    angle = 2.0 * math.pi * index / n
    offset: Array = START_OFFSET_M * np.array([math.cos(angle), math.sin(angle)])
    return base + offset


def benign_speed(benign_class: str) -> float:
    """BenignRoute has no speed field, so speed comes from the class."""
    return BENIGN_SPEED_MPS.get(benign_class, DEFAULT_BENIGN_SPEED_MPS)


def sensor_max_range_m(sensor_curves: SensorCurves, sensor_type: str) -> float:
    return float(_curve(sensor_curves, sensor_type).max_range_m())


def sensor_look_rate_hz(sensor_curves: SensorCurves, sensor_type: str) -> float:
    return float(_curve(sensor_curves, sensor_type).look_rate_hz)


def _curve(sensor_curves: SensorCurves, sensor_type: str) -> SensorCurve:
    if sensor_type not in sensor_curves.curves:
        raise KeyError(f"no sensor curve {sensor_type!r}; known: {sorted(sensor_curves.curves)}")
    return sensor_curves.curves[sensor_type]
