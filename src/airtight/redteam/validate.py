from __future__ import annotations

import math
from typing import TYPE_CHECKING

from airtight.redteam.config import RedTeamConfig
from airtight.redteam.geometry import (
    distance_to_boundary,
    in_bounds,
    path_inside_polygon,
    path_length,
)

if TYPE_CHECKING:
    from airtight.contracts import Site, Tactic


def validate(tactic: Tactic, site: Site, cfg: RedTeamConfig | None = None) -> list[str]:
    """Every reason the tactic is not playable on this site; empty means valid."""
    cfg = cfg or RedTeamConfig()
    errors: list[str] = []
    try:
        entry = site.entry(tactic.entry_id)
    except KeyError as e:
        return [str(e)]
    if distance_to_boundary(entry.position, site.perimeter) > cfg.perimeter_tol_m:
        errors.append(
            f"entry {entry.id!r} is {distance_to_boundary(entry.position, site.perimeter):.1f} m from the perimeter; must lie on it"
        )
    if not (cfg.speed_min_mps <= tactic.speed_mps <= cfg.speed_cap_mps):
        errors.append(
            f"speed {tactic.speed_mps:.2f} m/s outside [{cfg.speed_min_mps}, {cfg.speed_cap_mps}]"
        )
    if len(tactic.waypoints) > cfg.max_waypoints:
        errors.append(f"{len(tactic.waypoints)} waypoints exceeds max {cfg.max_waypoints}")
    last = tactic.waypoints[-1]
    if math.hypot(last.x - site.asset.x, last.y - site.asset.y) > cfg.asset_tol_m:
        errors.append(
            f"last waypoint ({last.x:.1f}, {last.y:.1f}) is not the asset ({site.asset.x:.1f}, {site.asset.y:.1f})"
        )
    path = [entry.position, *tactic.waypoints]
    if not all(in_bounds(p, site.bounds) for p in path):
        errors.append("a waypoint lies outside the site bounds")
    if not path_inside_polygon(path, site.perimeter, tol=cfg.perimeter_tol_m):
        errors.append("the path leaves the perimeter between the entry point and the asset")
    for a, b in zip(path, path[1:], strict=False):
        if math.hypot(b.x - a.x, b.y - a.y) < cfg.min_leg_m:
            errors.append(f"leg shorter than {cfg.min_leg_m} m at ({a.x:.1f}, {a.y:.1f})")
            break
    if tactic.decoy is not None:
        if not in_bounds(tactic.decoy.position, site.bounds):
            errors.append("decoy position lies outside the site bounds")
        lo, hi = cfg.decoy_lead_s
        if not (lo <= tactic.decoy.lead_time_s <= hi):
            errors.append(
                f"decoy lead {tactic.decoy.lead_time_s:.0f} s outside [{lo:.0f}, {hi:.0f}]"
            )
    if tactic.comms_event is not None:
        travel_s = path_length(path) / tactic.speed_mps
        if tactic.comms_event.t_s > travel_s:
            errors.append(
                f"comms event at {tactic.comms_event.t_s:.0f} s is after the intruder reaches the asset ({travel_s:.0f} s)"
            )
    if tactic.family == "decoy" and tactic.decoy is None:
        errors.append("decoy family requires a decoy")
    if tactic.family == "comms_cut" and tactic.comms_event is None:
        errors.append("comms_cut family requires a comms event")
    return errors


def is_valid(tactic: Tactic, site: Site, cfg: RedTeamConfig | None = None) -> bool:
    return not validate(tactic, site, cfg)
