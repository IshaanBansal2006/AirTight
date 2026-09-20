"""The only module in sim/ that reads ambiguous contract fields.

Contract models go in; plain floats and numpy arrays come out, so a contract change lands here
and nowhere else in the lane.

Time: t = 0 is the moment the intruder stands on its entry point, which is what the log
contract ("seconds since the intruder entered") and the stub runner both mean. Warm-up runs at
negative t on the same clock and is never logged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from airtight.sim.constants import DECOY_DURATION_S
from airtight.sim.geometry import polyline_length

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
_PD_CACHE: dict[int, tuple[object, np.ndarray, np.ndarray]] = {}


class BenignRouteSpec(NamedTuple):
    route_id: str
    cls: str
    points: Array  # (n, 2)
    arrivals_per_hour: float
    speed_mps: float


class EnergySpec(NamedTuple):
    endurance_s: float  # operating time from full charge to the charge threshold
    charge_time_s: float  # dock time from the threshold back to full
    offset_s: float  # how far into its cycle the agent already is at absolute time 0


class FixedSensorSpec(NamedTuple):
    sensor_id: str
    position: Array  # (2,)
    heading_rad: float  # the contract stores degrees
    sensor_type: str


class DecoySpec(NamedTuple):
    position: Array  # (2,)
    t_on: float  # episode clock; negative when the decoy leads the intruder
    t_off: float


def bounds(site: Site) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in metres."""
    xmin, ymin, xmax, ymax = site.bounds
    return (float(xmin), float(ymin), float(xmax), float(ymax))


def perimeter(site: Site) -> Array:
    """Perimeter vertices in order, shape (n, 2)."""
    return np.array([[p.x, p.y] for p in site.perimeter], dtype=np.float64)


def docks(site: Site) -> Array:
    """Dock positions in site order, shape (n, 2); (0, 2) when the site has none."""
    return np.array([[d.position.x, d.position.y] for d in site.docks], dtype=np.float64).reshape(
        -1, 2
    )


def assets(site: Site) -> Array:
    """Asset positions, shape (n, 2). The contract has one asset today."""
    return np.array([[site.asset.x, site.asset.y]], dtype=np.float64)


def intruder_path(site: Site, tactic: Tactic) -> Array:
    """Entry point then the tactic's waypoints, shape (n, 2). Same path as the stub runner."""
    pts = [site.entry(tactic.entry_id).position, *tactic.waypoints]
    return np.array([[p.x, p.y] for p in pts], dtype=np.float64)


def path_length_m(path: Array) -> float:
    return polyline_length(path)


def t_reach(site: Site, tactic: Tactic) -> float:
    """Seconds after entry at which the intruder reaches the asset."""
    return path_length_m(intruder_path(site, tactic)) / tactic.speed_mps


def t_cdp(site: Site, tactic: Tactic) -> float:
    """Critical detection point: t_reach minus the response time, clamped at 0 as in the stub."""
    return max(0.0, t_reach(site, tactic) - site.response_time_s)


def response_time_s(site: Site) -> float:
    return float(site.response_time_s)


def critical_radius_m(site: Site, v_ref_mps: float) -> float:
    """r_c = v_ref * response_time_s: the ring round an asset inside which a detection of an
    intruder moving at v_ref is already too late for the response to arrive."""
    return v_ref_mps * float(site.response_time_s)


def intruder_speed_mps(tactic: Tactic) -> float:
    return float(tactic.speed_mps)


def decoy_spec(tactic: Tactic) -> DecoySpec | None:
    """The tactic's decoy on the episode clock, or None.

    It switches on lead_time_s before the intruder enters, so t_on = -lead_time_s. The contract
    has no duration; it stays on for DECOY_DURATION_S.
    """
    if tactic.decoy is None:
        return None
    position = np.array([tactic.decoy.position.x, tactic.decoy.position.y], dtype=np.float64)
    t_on = -float(tactic.decoy.lead_time_s)
    return DecoySpec(position=position, t_on=t_on, t_off=t_on + DECOY_DURATION_S)


def agent_ids(fleet: FleetConfig) -> list[str]:
    """Agent ids in fleet order. An agent's index everywhere in sim/ is its position here."""
    return [a.id for a in fleet.agents]


def agent_speed_mps(fleet: FleetConfig, agent_id: str) -> float:
    return float(_agent(fleet, agent_id).speed_mps)


def agent_sensor_type(fleet: FleetConfig, agent_id: str) -> str:
    return _agent(fleet, agent_id).sensor_type


NO_CHARGE_REFERENCE_CYCLE_S = 3600.0


def agent_energy(fleet: FleetConfig, agent_id: str) -> EnergySpec | None:
    """The agent's charge cycle, or None if it never charges.

    The contract has no "never charges" field: endurance_s is required. The team's convention
    is charge_time_s == 0 (the contract says "0 for guards"). The offset is the contract's
    ChargePolicy.stagger_offsets_s, "initial phase offset", 0.0 when the agent is not listed.
    """
    agent = _agent(fleet, agent_id)
    if agent.charge_time_s <= 0:
        return None
    return EnergySpec(
        endurance_s=float(agent.endurance_s),
        charge_time_s=float(agent.charge_time_s),
        offset_s=float(fleet.charge_policy.stagger_offsets_s.get(agent_id, 0.0)),
    )


def reference_cycle_s(fleet: FleetConfig) -> float:
    """What Tactic.phase is a fraction of: "the fleet's charge cycle".

    Lane C defined it first (redteam/families.py charge_cycle_s) and that definition wins: the
    mean of endurance_s + charge_time_s over agents that charge, or 3600 s if none do. For a
    mixed fleet this is not the period of any one agent; it only turns a phase into a time.
    """
    cycles = [
        spec.endurance_s + spec.charge_time_s
        for spec in (agent_energy(fleet, a) for a in agent_ids(fleet))
        if spec is not None
    ]
    return float(np.mean(cycles)) if cycles else NO_CHARGE_REFERENCE_CYCLE_S


def tactic_phase(tactic: Tactic) -> float:
    """Fraction in [0, 1) of reference_cycle_s at which the intruder enters."""
    return float(tactic.phase)


def start_position(site: Site, fleet: FleetConfig, agent_id: str) -> Array:
    """Docks round-robin by fleet index, else the perimeter centroid, plus a small offset.

    The offset is START_OFFSET_M at angle 2*pi*index/n. Two agents on the exact same point tie
    everywhere in a Voronoi test and the higher index never gets a region. Dock capacity is
    ignored here.
    """
    ids = agent_ids(fleet)
    _agent(fleet, agent_id)
    index, n = ids.index(agent_id), len(ids)
    if site.docks:
        dock = site.docks[index % len(site.docks)].position
        base = np.array([dock.x, dock.y], dtype=np.float64)
    else:
        base = perimeter(site).mean(axis=0)
    angle = 2.0 * math.pi * index / n
    offset: Array = START_OFFSET_M * np.array([math.cos(angle), math.sin(angle)])
    return base + offset


def fixed_sensors(site: Site) -> list[FixedSensorSpec]:
    """The site's fixed sensors in site order, heading converted from degrees to radians."""
    return [
        FixedSensorSpec(
            sensor_id=s.id,
            position=np.array([s.position.x, s.position.y], dtype=np.float64),
            heading_rad=math.radians(s.heading_deg),
            sensor_type=s.sensor_type,
        )
        for s in site.fixed_sensors
    ]


def benign_speed(benign_class: str) -> float:
    """BenignRoute has no speed field, so speed comes from the class.

    The contract types BenignRoute.cls as a free string, not a closed Literal, so a site may
    name a class this table has never heard of. Those fall back to DEFAULT_BENIGN_SPEED_MPS
    (walking pace) rather than raising. If the contract ever closes the set, make this a KeyError.
    """
    return BENIGN_SPEED_MPS.get(benign_class, DEFAULT_BENIGN_SPEED_MPS)


def benign_routes(site: Site) -> list[BenignRouteSpec]:
    """Benign routes in site order. The contract has no per-route speed, so it comes from cls."""
    return [
        BenignRouteSpec(
            route_id=route.id,
            cls=route.cls,
            points=np.array([[p.x, p.y] for p in route.waypoints], dtype=np.float64),
            arrivals_per_hour=float(route.arrival_rate_per_hour),
            speed_mps=benign_speed(route.cls),
        )
        for route in site.benign_routes
    ]


def has_curve(sensor_curves: SensorCurves, sensor_type: str) -> bool:
    return sensor_type in sensor_curves.curves


def fp_classes(sensor_curves: SensorCurves, sensor_type: str) -> set[str]:
    """Benign classes this sensor has a true false-positive rate for."""
    return set(_curve(sensor_curves, sensor_type).pfa_per_look_by_class)


def sensor_max_range_m(sensor_curves: SensorCurves, sensor_type: str) -> float:
    return float(_curve(sensor_curves, sensor_type).max_range_m())


def sensor_look_rate_hz(sensor_curves: SensorCurves, sensor_type: str) -> float:
    return float(_curve(sensor_curves, sensor_type).look_rate_hz)


def sensor_footprint_radius_m(sensor_curves: SensorCurves, sensor_type: str) -> float:
    """Patrol footprint: the upper edge of the last range bin whose pd_per_look is > 0.

    Distinct from sensor_max_range_m, which is the last bin edge whatever its probability.
    """
    curve = _curve(sensor_curves, sensor_type)
    live = [edge for edge, pd in zip(curve.range_bins_m, curve.pd_per_look, strict=True) if pd > 0]
    if not live:
        raise ValueError(f"sensor {sensor_type!r} has pd_per_look == 0 in every range bin")
    return float(live[-1])


def pd_per_look(sensor_curves: SensorCurves, sensor_type: str, range_m: float) -> float:
    """Detection probability per look in the bin holding range_m; 0.0 beyond the last bin.

    Bins are upper edges, so a range exactly on an edge belongs to the bin that edge closes.
    """
    curve = _curve(sensor_curves, sensor_type)
    cached = _PD_CACHE.get(id(curve))
    if cached is None or cached[0] is not curve:
        cached = (
            curve,
            np.asarray(curve.range_bins_m, dtype=np.float64),
            np.asarray(curve.pd_per_look, dtype=np.float64),
        )
        _PD_CACHE[id(curve)] = cached
    _, bins, pd = cached
    i = int(np.searchsorted(bins, range_m, side="left"))
    return float(pd[i]) if i < bins.size else 0.0


def true_fp_per_look(sensor_curves: SensorCurves, sensor_type: str, cls: str) -> float:
    """The TRUE per-look false-positive rate for a benign class: one scalar, not per range bin."""
    table = _curve(sensor_curves, sensor_type).pfa_per_look_by_class
    if cls not in table:
        raise KeyError(
            f"sensor {sensor_type!r} has no false-positive rate for class {cls!r}; "
            f"known: {sorted(table)}"
        )
    return float(table[cls])


def sensor_fov_deg(sensor_curves: SensorCurves, sensor_type: str) -> float:
    """The contract's SensorCurve.fov_deg, a required field in (0, 360]."""
    return float(_curve(sensor_curves, sensor_type).fov_deg)


def _agent(fleet: FleetConfig, agent_id: str) -> AgentSpec:
    for agent in fleet.agents:
        if agent.id == agent_id:
            return agent
    raise ValueError(
        f"agent {agent_id!r} is not in fleet {fleet.name!r}; known: {agent_ids(fleet)}"
    )


def _curve(sensor_curves: SensorCurves, sensor_type: str) -> SensorCurve:
    if sensor_type not in sensor_curves.curves:
        raise KeyError(f"no sensor curve {sensor_type!r}; known: {sorted(sensor_curves.curves)}")
    return sensor_curves.curves[sensor_type]
