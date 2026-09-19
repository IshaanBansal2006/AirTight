from __future__ import annotations

import math

import numpy as np

from airtight.contracts import (
    XY,
    CommsEvent,
    Decoy,
    FleetConfig,
    SensorCurves,
    Site,
    Tactic,
    TacticFamily,
)
from airtight.redteam.config import RedTeamConfig
from airtight.redteam.coverage import CoverageMap, GeometryCoverage
from airtight.redteam.geometry import path_length, point_in_polygon, progress_along
from airtight.redteam.validate import validate

FAMILIES: tuple[TacticFamily, ...] = ("charging_window", "decoy", "blind_spot", "comms_cut")
_MAX_TRIES = 60


def charge_cycle_s(fleet: FleetConfig) -> float:
    """Mean endurance plus charge time over agents that charge; the phase parameter is a fraction of this."""
    cycles = [a.endurance_s + a.charge_time_s for a in fleet.agents if a.charge_time_s > 0]
    return float(np.mean(cycles)) if cycles else 3600.0


def new_id(family: TacticFamily, rng: np.random.Generator) -> str:
    return f"{family}-{int(rng.integers(0, 10**9)):09d}"


def sample_tactic(
    family: TacticFamily,
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    rng: np.random.Generator,
    cfg: RedTeamConfig | None = None,
    coverage: CoverageMap | None = None,
) -> Tactic:
    """Rejection-sample one valid tactic of the family. Raises if the site admits none."""
    cfg = cfg or RedTeamConfig()
    if family == "blind_spot" and coverage is None:
        coverage = GeometryCoverage(cfg.coverage_cell_m, cfg.dock_halo_m).coverage(site, curves)
    last_errors: list[str] = []
    for _ in range(_MAX_TRIES):
        tactic = _draw(family, site, fleet, rng, cfg, coverage)
        last_errors = validate(tactic, site, cfg)
        if not last_errors:
            return tactic
    raise RuntimeError(
        f"could not sample a valid {family} tactic on site {site.name!r} in {_MAX_TRIES} tries; last errors: {last_errors}"
    )


def _draw(
    family: TacticFamily,
    site: Site,
    fleet: FleetConfig,
    rng: np.random.Generator,
    cfg: RedTeamConfig,
    coverage: CoverageMap | None,
) -> Tactic:
    entry = site.entry_points[int(rng.integers(len(site.entry_points)))]
    phase = float(rng.uniform(0.0, 1.0))
    speed = float(rng.uniform(cfg.speed_min_mps, cfg.speed_cap_mps))
    if family == "blind_spot":
        assert coverage is not None
        waypoints = _blind_spot_path(site, entry.position, coverage, rng, cfg)
    else:
        waypoints = _bent_path(site, entry.position, rng, cfg)
    decoy = _draw_decoy(site, entry.id, rng, cfg) if family == "decoy" else None
    comms = None
    if family == "comms_cut":
        travel_s = path_length([entry.position, *waypoints]) / speed
        comms = CommsEvent(t_s=float(rng.uniform(0.0, 0.6 * travel_s)), kind="cut_base_link")
    return Tactic(
        id=new_id(family, rng),
        family=family,
        entry_id=entry.id,
        phase=phase,
        speed_mps=speed,
        waypoints=waypoints,
        decoy=decoy,
        comms_event=comms,
        origin="random",
    )


def _bent_path(site: Site, start: XY, rng: np.random.Generator, cfg: RedTeamConfig) -> list[XY]:
    """Entry to asset with one elbow displaced sideways, so straight-line and flanking approaches both appear."""
    asset = site.asset
    mid = XY(x=(start.x + asset.x) / 2, y=(start.y + asset.y) / 2)
    length = math.hypot(asset.x - start.x, asset.y - start.y)
    if length == 0.0:
        return [asset]
    nx, ny = -(asset.y - start.y) / length, (asset.x - start.x) / length
    for _ in range(_MAX_TRIES):
        offset = float(rng.normal(0.0, 0.25 * length))
        elbow = XY(x=mid.x + nx * offset, y=mid.y + ny * offset)
        if point_in_polygon(elbow, site.perimeter):
            return [elbow, asset]
    return [asset]


def _blind_spot_path(
    site: Site, start: XY, coverage: CoverageMap, rng: np.random.Generator, cfg: RedTeamConfig
) -> list[XY]:
    """Thread the approach through unwatched cells, visiting them in order of progress toward the asset."""
    low = coverage.low_cells(site)
    if not low:
        return _bent_path(site, start, rng, cfg)
    k = int(rng.integers(1, min(3, cfg.max_waypoints - 1) + 1))
    picks = [low[int(i)] for i in rng.choice(len(low), size=min(k, len(low)), replace=False)]
    picks.sort(key=lambda p: progress_along(p, start, site.asset))
    picks = [p for p in picks if 0.0 < progress_along(p, start, site.asset) < 1.0]
    return [*picks, site.asset] if picks else _bent_path(site, start, rng, cfg)


def _draw_decoy(site: Site, entry_id: str, rng: np.random.Generator, cfg: RedTeamConfig) -> Decoy:
    """A decoy appears near a different entry point, just inside the fence, before the real intruder enters."""
    others = [e for e in site.entry_points if e.id != entry_id] or list(site.entry_points)
    e = others[int(rng.integers(len(others)))]
    lo, hi = cfg.decoy_lead_s
    for _ in range(_MAX_TRIES):
        ang = float(rng.uniform(0.0, 2 * math.pi))
        p = XY(
            x=e.position.x + cfg.decoy_offset_m * math.cos(ang),
            y=e.position.y + cfg.decoy_offset_m * math.sin(ang),
        )
        if point_in_polygon(p, site.perimeter):
            return Decoy(position=p, lead_time_s=float(rng.uniform(lo, hi)))
    return Decoy(position=site.asset, lead_time_s=float(rng.uniform(lo, hi)))


def perturb(
    tactic: Tactic, site: Site, rng: np.random.Generator, cfg: RedTeamConfig | None = None
) -> Tactic:
    """Local move in tactic space; returns the parent unchanged if no valid neighbour is found."""
    cfg = cfg or RedTeamConfig()
    for _ in range(_MAX_TRIES):
        phase = float((tactic.phase + rng.normal(0.0, 0.05)) % 1.0)
        speed = float(
            np.clip(
                tactic.speed_mps * (1.0 + rng.normal(0.0, 0.1)),
                cfg.speed_min_mps,
                cfg.speed_cap_mps,
            )
        )
        pts = [
            XY(
                x=p.x + float(rng.normal(0.0, cfg.waypoint_jitter_m)),
                y=p.y + float(rng.normal(0.0, cfg.waypoint_jitter_m)),
            )
            for p in tactic.waypoints[:-1]
        ] + [tactic.waypoints[-1]]
        decoy = None
        if tactic.decoy is not None:
            lo, hi = cfg.decoy_lead_s
            decoy = Decoy(
                position=XY(
                    x=tactic.decoy.position.x + float(rng.normal(0.0, cfg.waypoint_jitter_m)),
                    y=tactic.decoy.position.y + float(rng.normal(0.0, cfg.waypoint_jitter_m)),
                ),
                lead_time_s=float(
                    np.clip(tactic.decoy.lead_time_s + rng.normal(0.0, 20.0), lo, hi)
                ),
            )
        comms = None
        if tactic.comms_event is not None:
            comms = CommsEvent(
                t_s=float(max(0.0, tactic.comms_event.t_s + rng.normal(0.0, 10.0))),
                kind=tactic.comms_event.kind,
            )
        child = tactic.model_copy(
            update={
                "id": new_id(tactic.family, rng),
                "phase": phase,
                "speed_mps": speed,
                "waypoints": pts,
                "decoy": decoy,
                "comms_event": comms,
                "origin": "search",
            }
        )
        if not validate(child, site, cfg):
            return child
    return tactic
