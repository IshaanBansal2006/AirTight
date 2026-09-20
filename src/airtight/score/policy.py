"""The defender's free policy space: what may be optimised at no hardware cost.

A policy is the three mixture gains of the patrol weight, its length scale, the three patrol
controller parameters, a charge offset for every agent that charges, and a dock for every
agent that charges. Applying one to a fleet changes only ChargePolicy.stagger_offsets_s and
the fleet's name; everything else goes into EpisodeParams, which reaches run_episode through
AIRTIGHT_PARAMS_JSON. Agents, cost, site, sensors and the intruder are never touched.

The weight base stays at its default: the controller ranks cells by a product in which the
weight enters linearly, so only the ratios of the gains to the base matter.

Dock capacity: the engine ignores it, but a sampled assignment respects the site's capacities
whenever the agents that charge fit, and spreads the overflow evenly when they do not.

All randomness comes from the generator passed in, which the caller keys explicitly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from airtight.sim.episode import EpisodeParams, official_params, params_to_mapping

if TYPE_CHECKING:
    import numpy as np

    from airtight.contracts import FleetConfig, Site

GAIN_RANGE = (0.1, 5.0)
BAND_RANGE = (0.1, 3.0)
SCALE_RANGE_M = (10.0, 80.0)
D0_RANGE_M = (15.0, 200.0)
RETARGET_RANGE_S = (4.0, 30.0)
TOP_FRACTION_RANGE = (0.01, 0.3)
P_ZERO_GAIN = 0.3
PERTURB_SIGMA = 0.25
PERTURB_OFFSET_FRAC = 0.05
FLOAT_FIELDS = (
    "asset_gain",
    "entry_gain",
    "band_gain",
    "weight_scale_m",
    "d0_m",
    "retarget_period_s",
    "top_fraction",
)
_RANGES = {
    "asset_gain": GAIN_RANGE,
    "entry_gain": GAIN_RANGE,
    "band_gain": BAND_RANGE,
    "weight_scale_m": SCALE_RANGE_M,
    "d0_m": D0_RANGE_M,
    "retarget_period_s": RETARGET_RANGE_S,
    "top_fraction": TOP_FRACTION_RANGE,
}
_GAINS = ("asset_gain", "entry_gain", "band_gain")


@dataclass(frozen=True)
class Policy:
    asset_gain: float
    entry_gain: float
    band_gain: float
    weight_scale_m: float
    d0_m: float
    retarget_period_s: float
    top_fraction: float
    offsets: tuple[tuple[str, float], ...]  # (agent id, seconds), agents that charge, sorted
    docks: tuple[tuple[str, str], ...]  # (agent id, dock id), sorted; () is round-robin
    origin: str = "random"  # where the candidate came from; never affects a simulation

    def params(self) -> EpisodeParams:
        return dataclasses.replace(
            official_params(),
            weight_mode="mix",
            asset_gain=self.asset_gain,
            entry_gain=self.entry_gain,
            band_gain=self.band_gain,
            weight_scale_m=self.weight_scale_m,
            d0_m=self.d0_m,
            retarget_period_s=self.retarget_period_s,
            top_fraction=self.top_fraction,
            dock_assignment=self.docks,
        )

    def digest(self) -> str:
        body = {k: v for k, v in dataclasses.asdict(self).items() if k != "origin"}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]

    def apply(self, fleet: FleetConfig, name: str | None = None) -> FleetConfig:
        """The fleet with this policy's charge offsets. Same agents, same cost."""
        offsets = {agent: value for agent, value in self.offsets if value > 0}
        charge = fleet.charge_policy.model_copy(update={"stagger_offsets_s": offsets})
        new = fleet.model_copy(
            update={"name": name or f"{fleet.name}__p{self.digest()}", "charge_policy": charge}
        )
        return type(fleet).model_validate(new.model_dump())

    def params_json(self) -> dict[str, object]:
        """What to put in the AIRTIGHT_PARAMS_JSON file."""
        return params_to_mapping(self.params())

    def describe(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "offsets": dict(self.offsets),
            "docks": dict(self.docks),
        }


def chargers(fleet: FleetConfig) -> list[tuple[str, float]]:
    """(agent id, cycle seconds) of every agent that charges, in fleet order."""
    return [
        (a.id, float(a.endurance_s + a.charge_time_s)) for a in fleet.agents if a.charge_time_s > 0
    ]


def named_policy(
    fleet: FleetConfig,
    mode: str,
    stagger_fraction: float = 0.0,
    go2_half_cycle: bool = False,
    origin: str | None = None,
) -> Policy:
    """The named weight modes and the fix loop's offsets, as points of the policy space.
    Drones are staggered by stagger_fraction of even spacing; the Go2 sits at 0 or half its
    cycle. These seed every search, so the optimiser can never do worse than a known policy
    on its search seeds."""
    base = official_params()
    gains = {
        "asset": (base.asset_gain, 0.0, 0.0),
        "uniform": (0.0, 0.0, 0.0),
        "band": (0.0, 0.0, base.asset_gain),
    }[mode]
    drones = [a for a in fleet.agents if a.type == "drone" and a.charge_time_s > 0]
    offsets: dict[str, float] = {agent: 0.0 for agent, _ in chargers(fleet)}
    for i, drone in enumerate(drones):
        cycle = drone.endurance_s + drone.charge_time_s
        offsets[drone.id] = round(stagger_fraction * i * cycle / len(drones), 1)
    if go2_half_cycle:
        for agent in fleet.agents:
            if agent.type == "go2" and agent.charge_time_s > 0:
                offsets[agent.id] = round((agent.endurance_s + agent.charge_time_s) / 2.0, 1)
    return Policy(
        *gains,
        weight_scale_m=base.weight_scale_m,
        d0_m=base.d0_m,
        retarget_period_s=base.retarget_period_s,
        top_fraction=base.top_fraction,
        offsets=tuple(sorted(offsets.items())),
        docks=(),
        origin=origin or f"named:{mode}:s{stagger_fraction:g}:g{'half' if go2_half_cycle else '0'}",
    )


def seed_policies(fleet: FleetConfig) -> list[Policy]:
    out = []
    for mode in ("asset", "uniform", "band"):
        for fraction in (0.0, 1.0):
            for half in (False, True):
                out.append(named_policy(fleet, mode, fraction, half))
    return list({p.digest(): p for p in out}.values())


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_docks(
    site: Site, fleet: FleetConfig, rng: np.random.Generator
) -> tuple[tuple[str, str], ...]:
    """A random dock for every agent that charges, within capacity when they fit."""
    agents = [agent for agent, _ in chargers(fleet)]
    if not site.docks or not agents:
        return ()
    slots = [d.id for d in site.docks for _ in range(max(int(d.capacity), 1))]
    while len(slots) < len(agents):
        slots += [d.id for d in site.docks]
    order = rng.permutation(len(slots))
    picked = [slots[int(i)] for i in order[: len(agents)]]
    return tuple(sorted(zip(agents, picked, strict=True)))


def sample_policy(site: Site, fleet: FleetConfig, rng: np.random.Generator) -> Policy:
    values: dict[str, float] = {}
    for name in FLOAT_FIELDS:
        lo, hi = _RANGES[name]
        zero = name in _GAINS and rng.random() < P_ZERO_GAIN
        values[name] = 0.0 if zero else round(_log_uniform(rng, lo, hi), 4)
    offsets: dict[str, float] = {}
    charging = chargers(fleet)
    even = rng.random() < 0.5
    fraction = float(rng.uniform(0.5, 1.0))
    start = rng.permutation(len(charging))
    for rank, (agent, cycle) in zip(start, charging, strict=True):
        if even:
            value = fraction * int(rank) * cycle / len(charging) + rng.normal(0.0, 0.02 * cycle)
        else:
            value = rng.uniform(0.0, cycle)
        offsets[agent] = round(float(value) % cycle, 1) % cycle
    docks = sample_docks(site, fleet, rng) if rng.random() < 0.5 else ()
    return Policy(**values, offsets=tuple(sorted(offsets.items())), docks=docks, origin="random")


def perturb(site: Site, fleet: FleetConfig, policy: Policy, rng: np.random.Generator) -> Policy:
    """A neighbour: every number nudged, a zero gain sometimes switched on, one dock moved."""
    values: dict[str, float] = {}
    for name in FLOAT_FIELDS:
        lo, hi = _RANGES[name]
        value = float(getattr(policy, name))
        if name in _GAINS and rng.random() < 0.1:
            value = 0.0 if value > 0 else _log_uniform(rng, lo, hi)
        elif value > 0:
            value = min(max(value * math.exp(rng.normal(0.0, PERTURB_SIGMA)), lo), hi)
        values[name] = round(value, 4)
    cycles = dict(chargers(fleet))
    offsets = {
        agent: round(
            float(value + rng.normal(0.0, PERTURB_OFFSET_FRAC * cycles[agent])) % cycles[agent], 1
        )
        % cycles[agent]
        for agent, value in policy.offsets
    }
    docks = policy.docks
    if site.docks and cycles and rng.random() < 0.2:
        moved = dict(docks) if docks else {}
        agent = sorted(cycles)[int(rng.integers(len(cycles)))]
        moved[agent] = site.docks[int(rng.integers(len(site.docks)))].id
        docks = tuple(sorted(moved.items()))
    return Policy(
        **values,
        offsets=tuple(sorted(offsets.items())),
        docks=docks,
        origin=f"near:{policy.digest()}",
    )
