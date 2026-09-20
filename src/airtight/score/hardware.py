"""The priced hardware space of the optimisation campaign.

Three things can be bought: drones (one to six), a battery-swap upgrade for every drone dock,
and a fixed camera at every entry no existing fixed sensor covers. Nothing here simulates. The
site's geometry, sensor curves, intruder model, response time and benign traffic are never
changed: the only site edit is appending purchasable cameras to site.fixed_sensors.

Prices for the two purchasable upgrades live in a JSON file (data/campaign/costs.json) that
load_costs creates from DEFAULT_COSTS, in the style of the scenario's costs.md: hardware price
amortised over three years of 24/7 availability plus upkeep, each marked "verify" until someone
checks the source.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from airtight.contracts import FixedSensor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from airtight.contracts import FleetConfig, SensorCurve, SensorCurves, Site

HARDWARE_DRONES = range(1, 7)
SWAP_CHARGE_TIME_S = 180.0

AMORTISATION_HOURS = 26280
AMORTISATION_YEARS = 3
SWAP_DOCK_HARDWARE_USD = 18000
SWAP_DOCK_UPKEEP_USD_PER_YEAR = 5000
ENTRY_CAMERA_HARDWARE_USD = 4000
ENTRY_CAMERA_UPKEEP_USD_PER_YEAR = 2200

SWAP_COST_KEY = "swap_dock_per_drone_usd_per_hour"
CAMERA_COST_KEY = "entry_camera_usd_per_hour"
COST_UNITS = "USD per hour"
COST_STATUSES = ("verify", "verified")
CAMERA_ID_PREFIX = "cam_entry_"


def _amortised(hardware_usd: int, upkeep_usd_per_year: int) -> float:
    total = hardware_usd + AMORTISATION_YEARS * upkeep_usd_per_year
    return round(total / AMORTISATION_HOURS, 2)


def _arithmetic(hardware_usd: int, upkeep_usd_per_year: int) -> str:
    return (
        f"({hardware_usd:,} + {AMORTISATION_YEARS} x {upkeep_usd_per_year:,}) / "
        f"{AMORTISATION_HOURS:,} h = {_amortised(hardware_usd, upkeep_usd_per_year):.2f} $/h"
    )


DEFAULT_COSTS: dict[str, dict[str, Any]] = {
    SWAP_COST_KEY: {
        "value": _amortised(SWAP_DOCK_HARDWARE_USD, SWAP_DOCK_UPKEEP_USD_PER_YEAR),
        "units": COST_UNITS,
        "source": (
            "Battery-swap dock instead of a charge-in-place dock, per drone: about "
            f"${SWAP_DOCK_HARDWARE_USD:,} hardware premium for the swap mechanism, plus about "
            f"${SWAP_DOCK_UPKEEP_USD_PER_YEAR:,}/year for the extra battery packs it cycles and "
            "servicing of the mechanism, amortised over three years of 24/7 availability: "
            f"{_arithmetic(SWAP_DOCK_HARDWARE_USD, SWAP_DOCK_UPKEEP_USD_PER_YEAR)}; verify "
            "against a quote for a swap-capable dock and its battery replacement schedule."
        ),
        "status": "verify",
    },
    CAMERA_COST_KEY: {
        "value": _amortised(ENTRY_CAMERA_HARDWARE_USD, ENTRY_CAMERA_UPKEEP_USD_PER_YEAR),
        "units": COST_UNITS,
        "source": (
            "Fixed PoE analytics camera at one entry: about "
            f"${ENTRY_CAMERA_HARDWARE_USD:,} for the camera, pole or bracket, cable run and "
            f"installation labour, plus about ${ENTRY_CAMERA_UPKEEP_USD_PER_YEAR:,}/year for the "
            "VMS and analytics licence, storage, cleaning and monitoring share, amortised over "
            "three years of 24/7 availability: "
            f"{_arithmetic(ENTRY_CAMERA_HARDWARE_USD, ENTRY_CAMERA_UPKEEP_USD_PER_YEAR)}; verify "
            "against an installer quote and a VMS per-channel licence price."
        ),
        "status": "verify",
    },
}


@dataclass(frozen=True)
class HardwareSpec:
    """One point of the hardware space."""

    n_drones: int
    swap_docks: bool
    entry_cameras: bool

    @property
    def name(self) -> str:
        swap = "swap" if self.swap_docks else "std"
        cams = "cams" if self.entry_cameras else "nocams"
        return f"d{self.n_drones}_{swap}_{cams}"


def all_specs() -> list[HardwareSpec]:
    """Every combination, ordered by drone count, then swap docks, then entry cameras."""
    return [
        HardwareSpec(n, swap, cams)
        for n in HARDWARE_DRONES
        for swap in (False, True)
        for cams in (False, True)
    ]


def build_fleet(baseline: FleetConfig, spec: HardwareSpec) -> FleetConfig:
    """spec.n_drones clones of the baseline's first drone, then every other agent unchanged.

    Charge offsets are cleared: the policy layer sets them later for the fleet it is given.
    """
    drones = [a for a in baseline.agents if a.type == "drone"]
    if not drones:
        raise ValueError(f"baseline fleet {baseline.name!r} has no drone to clone")
    if spec.n_drones < 1:
        raise ValueError(f"n_drones must be at least 1, got {spec.n_drones}")
    others = [a for a in baseline.agents if a.type != "drone"]
    update: dict[str, Any] = {}
    if spec.swap_docks:
        update["charge_time_s"] = SWAP_CHARGE_TIME_S
    clones = [
        drones[0].model_copy(update={**update, "id": f"drone_{i}"})
        for i in range(1, spec.n_drones + 1)
    ]
    taken = {a.id for a in others} & {c.id for c in clones}
    if taken:
        raise ValueError(f"non-drone agents already use the drone ids {sorted(taken)}")
    policy = baseline.charge_policy.model_copy(update={"stagger_offsets_s": {}})
    return baseline.model_copy(
        update={"name": spec.name, "agents": [*clones, *others], "charge_policy": policy}
    )


def _footprint_radius_m(curve: SensorCurve) -> float:
    """The upper edge of the last range bin whose pd_per_look is > 0."""
    live = [edge for edge, pd in zip(curve.range_bins_m, curve.pd_per_look, strict=True) if pd > 0]
    if not live:
        raise ValueError(f"sensor {curve.sensor_type!r} has pd_per_look == 0 in every range bin")
    return float(live[-1])


def _curve(curves: SensorCurves, sensor_type: str) -> SensorCurve:
    if sensor_type not in curves.curves:
        raise ValueError(f"no sensor curve for {sensor_type!r}; known: {sorted(curves.curves)}")
    return curves.curves[sensor_type]


def _covers(sensor: FixedSensor, curve: SensorCurve, x: float, y: float, require_fov: bool) -> bool:
    dx = x - sensor.position.x
    dy = y - sensor.position.y
    if math.hypot(dx, dy) > _footprint_radius_m(curve):
        return False
    if not require_fov or curve.fov_deg >= 360.0 or (dx == 0 and dy == 0):
        return True
    off_axis = math.degrees(math.atan2(dy, dx)) - sensor.heading_deg
    off_axis = (off_axis + 180.0) % 360.0 - 180.0
    return abs(off_axis) <= curve.fov_deg / 2.0


def uncovered_entries(site: Site, curves: SensorCurves, require_fov: bool = True) -> list[str]:
    """Entry ids, in site order, whose entry point no existing fixed sensor covers.

    Covered means within the sensor's footprint radius and inside its field-of-view wedge
    centred on heading_deg (0 = +x, 90 = +y). require_fov=False keeps only the range test.
    """
    out = []
    for entry in site.entry_points:
        x, y = entry.position.x, entry.position.y
        if not any(
            _covers(s, _curve(curves, s.sensor_type), x, y, require_fov) for s in site.fixed_sensors
        ):
            out.append(entry.id)
    return out


def build_site(
    site: Site, curves: SensorCurves, spec: HardwareSpec, require_fov: bool = True
) -> Site:
    """The site, plus one camera facing the asset at every uncovered entry if the spec buys them."""
    if not spec.entry_cameras:
        return site
    if not site.fixed_sensors:
        raise ValueError(f"site {site.name!r} has no fixed sensor to copy the camera type from")
    types = sorted({s.sensor_type for s in site.fixed_sensors})
    if len(types) != 1:
        raise ValueError(f"site {site.name!r} mixes fixed sensor types {types}; cannot pick one")
    sensor_type = types[0]
    _curve(curves, sensor_type)
    added = []
    for entry_id in uncovered_entries(site, curves, require_fov):
        position = site.entry(entry_id).position
        heading = math.degrees(math.atan2(site.asset.y - position.y, site.asset.x - position.x))
        added.append(
            FixedSensor(
                id=f"{CAMERA_ID_PREFIX}{entry_id}",
                position=position,
                sensor_type=sensor_type,
                heading_deg=heading % 360.0,
            )
        )
    variant = site.model_copy(update={"fixed_sensors": [*site.fixed_sensors, *added]})
    return type(site).model_validate(variant.model_dump())


def load_costs(path: Path) -> dict[str, float]:
    """Price per purchasable item. Writes DEFAULT_COSTS first if the file does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_COSTS, indent=2) + "\n")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of cost entries")
    out: dict[str, float] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: cost entry {name!r} must be an object")
        value = entry.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
            raise ValueError(f"{path}: {name!r} needs a value >= 0, got {value!r}")
        source = entry.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}: {name!r} needs a non-empty source")
        if entry.get("status") not in COST_STATUSES:
            raise ValueError(
                f"{path}: {name!r} status must be one of {COST_STATUSES}, "
                f"got {entry.get('status')!r}"
            )
        out[str(name)] = float(value)
    missing = sorted(set(DEFAULT_COSTS) - set(out))
    if missing:
        raise ValueError(f"{path}: missing cost entries {missing}")
    return out


def cost_per_hour(
    fleet: FleetConfig,
    site: Site,
    base_site: Site,
    spec: HardwareSpec,
    costs: Mapping[str, float],
) -> float:
    """Fleet cost, plus the swap upgrade per drone, plus every camera added to base_site."""
    total = fleet.cost_per_hour()
    if spec.swap_docks:
        total += costs[SWAP_COST_KEY] * spec.n_drones
    base_ids = {s.id for s in base_site.fixed_sensors}
    n_added = sum(1 for s in site.fixed_sensors if s.id not in base_ids)
    if n_added:
        total += costs[CAMERA_COST_KEY] * n_added
    return total


def write_variants(
    out_dir: Path,
    baseline: FleetConfig,
    site: Site,
    curves: SensorCurves,
    specs: Sequence[HardwareSpec],
    require_fov: bool = True,
) -> dict[str, tuple[Path, Path]]:
    """Write <out_dir>/fleets/<name>.json and <out_dir>/sites/<name>.json for every spec."""
    fleets_dir = out_dir / "fleets"
    sites_dir = out_dir / "sites"
    fleets_dir.mkdir(parents=True, exist_ok=True)
    sites_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[Path, Path]] = {}
    for spec in specs:
        fleet_path = fleets_dir / f"{spec.name}.json"
        site_path = sites_dir / f"{spec.name}.json"
        fleet_path.write_text(build_fleet(baseline, spec).model_dump_json(indent=2) + "\n")
        site_path.write_text(
            build_site(site, curves, spec, require_fov).model_dump_json(indent=2) + "\n"
        )
        out[spec.name] = (fleet_path, site_path)
    return out
