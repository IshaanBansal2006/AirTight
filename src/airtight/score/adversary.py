"""A stronger stand-in adversary, all of it inside lane C's tactic limits.

Three sources of tactics, and the tools to keep them honest:

- strong_grid: every entry, straight to the asset, at both ends of the allowed speed range and
  at a fine grid of phases.
- random_tactics: seeded multi-waypoint paths drawn inside the perimeter.
- load_tactic_files: whatever lane C exported, parsed generically and never trusted blindly.

Every tactic is checked by validate_tactic, which re-implements the geometric checks of lane C's
airtight.redteam.validate without importing it: score/ stays independent of lane C's package.
The limits come from the scenario's redteam_config.json and are never typed here. On top of
lane C's limits a tactic must reach the asset within MAX_T_REACH_S, so an episode with a 120 s
task time still ends inside the 900 s benign window.

Nothing here simulates. keep_most_harmful only ranks tactics by detection numbers somebody else
measured.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError

from airtight.contracts import XY, Site, Tactic

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from airtight.contracts import TacticFamily, TacticOrigin

REDTEAM_CONFIG = "redteam_config.json"
MAX_T_REACH_S = 760.0
SPEED_EPS = 1e-9
SAMPLES_PER_LEG = 8
SNAP_PROBABILITY = 0.25
COORD_DECIMALS = 2
MAX_POINT_DRAWS = 10_000
MAX_TACTIC_DRAWS_PER_TACTIC = 1_000
STRONG_FAMILY: TacticFamily = "charging_window"
STRONG_ORIGIN: TacticOrigin = "hand"
RANDOM_FAMILY: TacticFamily = "blind_spot"
RANDOM_ORIGIN: TacticOrigin = "search"
TACTIC_MARKERS = ("entry_id", "waypoints")


@dataclass(frozen=True)
class Limits:
    speed_min_mps: float
    speed_cap_mps: float
    max_waypoints: int
    perimeter_tol_m: float
    asset_tol_m: float
    min_leg_m: float


def load_limits(scenario_dir: Path) -> Limits:
    """Lane C's tactic limits from the scenario's redteam_config.json."""
    path = scenario_dir / REDTEAM_CONFIG
    if not path.is_file():
        raise ValueError(f"no {REDTEAM_CONFIG} in {scenario_dir}: the tactic limits are unknown")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a JSON object of limits")
    names = [f.name for f in Limits.__dataclass_fields__.values()]
    missing = [name for name in names if name not in raw]
    if missing:
        raise ValueError(f"{path} is missing the limits {missing}")
    try:
        limits = Limits(
            speed_min_mps=float(raw["speed_min_mps"]),
            speed_cap_mps=float(raw["speed_cap_mps"]),
            max_waypoints=int(raw["max_waypoints"]),
            perimeter_tol_m=float(raw["perimeter_tol_m"]),
            asset_tol_m=float(raw["asset_tol_m"]),
            min_leg_m=float(raw["min_leg_m"]),
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path} holds a limit that is not a number: {e}") from e
    if not 0.0 < limits.speed_min_mps <= limits.speed_cap_mps:
        raise ValueError(
            f"{path}: need 0 < speed_min_mps <= speed_cap_mps, got "
            f"{limits.speed_min_mps} and {limits.speed_cap_mps}"
        )
    if limits.max_waypoints < 1:
        raise ValueError(f"{path}: max_waypoints must be at least 1, got {limits.max_waypoints}")
    return limits


def _point_in_polygon(p: XY, polygon: Sequence[XY]) -> bool:
    """Ray casting, the same rule lane C uses: an odd number of edge crossings means inside."""
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if (a.y > p.y) != (b.y > p.y):
            x_at_y = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if p.x < x_at_y:
                inside = not inside
    return inside


def _distance_to_segment(p: XY, a: XY, b: XY) -> float:
    abx, aby = b.x - a.x, b.y - a.y
    length_sq = abx * abx + aby * aby
    if length_sq == 0.0:
        return math.hypot(p.x - a.x, p.y - a.y)
    t = min(1.0, max(0.0, ((p.x - a.x) * abx + (p.y - a.y) * aby) / length_sq))
    return math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby))


def _distance_to_boundary(p: XY, polygon: Sequence[XY]) -> float:
    n = len(polygon)
    return min(_distance_to_segment(p, polygon[i], polygon[(i + 1) % n]) for i in range(n))


def _inside(p: XY, polygon: Sequence[XY], tol: float) -> bool:
    return _point_in_polygon(p, polygon) or _distance_to_boundary(p, polygon) <= tol


def _in_bounds(p: XY, bounds: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = bounds
    return xmin <= p.x <= xmax and ymin <= p.y <= ymax


def _legs(path: Sequence[XY]) -> list[tuple[XY, XY]]:
    return list(zip(path, path[1:], strict=False))


def _leg_samples(a: XY, b: XY) -> list[XY]:
    """SAMPLES_PER_LEG points strictly between a and b, as lane C samples a leg."""
    steps = SAMPLES_PER_LEG + 1
    return [
        XY(x=a.x + (b.x - a.x) * k / steps, y=a.y + (b.y - a.y) * k / steps)
        for k in range(1, steps)
    ]


def _path_length(path: Sequence[XY]) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in _legs(path))


def t_reach_s(site: Site, tactic: Tactic) -> float:
    """Seconds from the entry point to the last waypoint: path length over speed."""
    path = [site.entry(tactic.entry_id).position, *tactic.waypoints]
    return _path_length(path) / tactic.speed_mps


def validate_tactic(site: Site, limits: Limits, tactic: Tactic) -> list[str]:
    """Every reason the tactic is outside the limits; an empty list means valid."""
    try:
        entry = site.entry(tactic.entry_id)
    except KeyError:
        known = [e.id for e in site.entry_points]
        return [f"unknown entry id {tactic.entry_id!r}; the site has {known}"]
    reasons: list[str] = []
    perimeter = site.perimeter
    tol = limits.perimeter_tol_m
    entry_gap = _distance_to_boundary(entry.position, perimeter)
    if entry_gap > tol:
        reasons.append(f"entry {entry.id!r} is {entry_gap:.1f} m from the perimeter")
    lo, hi = limits.speed_min_mps, limits.speed_cap_mps
    if not (lo - SPEED_EPS <= tactic.speed_mps <= hi + SPEED_EPS):
        reasons.append(f"speed {tactic.speed_mps:.4f} m/s outside [{lo}, {hi}]")
    if len(tactic.waypoints) > limits.max_waypoints:
        reasons.append(f"{len(tactic.waypoints)} waypoints exceeds max {limits.max_waypoints}")
    last = tactic.waypoints[-1]
    asset_gap = math.hypot(last.x - site.asset.x, last.y - site.asset.y)
    if asset_gap > limits.asset_tol_m:
        reasons.append(
            f"last waypoint ({last.x:.1f}, {last.y:.1f}) is {asset_gap:.1f} m from the asset "
            f"({site.asset.x:.1f}, {site.asset.y:.1f})"
        )
    path = [entry.position, *tactic.waypoints]
    if not all(_in_bounds(p, site.bounds) for p in path):
        reasons.append("a waypoint lies outside the site bounds")
    outside = [p for p in tactic.waypoints if not _inside(p, perimeter, tol)]
    if outside:
        p = outside[0]
        reasons.append(
            f"waypoint ({p.x:.1f}, {p.y:.1f}) is outside the perimeter by more than {tol} m"
        )
    elif not all(_inside(q, perimeter, tol) for a, b in _legs(path) for q in _leg_samples(a, b)):
        reasons.append("the path leaves the perimeter between two waypoints")
    for a, b in _legs(path):
        if math.hypot(b.x - a.x, b.y - a.y) < limits.min_leg_m:
            reasons.append(f"leg shorter than {limits.min_leg_m} m at ({a.x:.1f}, {a.y:.1f})")
            break
    reach = _path_length(path) / tactic.speed_mps
    if reach > MAX_T_REACH_S:
        reasons.append(f"t_reach {reach:.1f} s exceeds {MAX_T_REACH_S} s")
    if tactic.decoy is not None and not _in_bounds(tactic.decoy.position, site.bounds):
        reasons.append("decoy position lies outside the site bounds")
    if tactic.comms_event is not None and tactic.comms_event.t_s > reach:
        reasons.append(
            f"comms event at {tactic.comms_event.t_s:.0f} s is after the intruder reaches the "
            f"asset ({reach:.0f} s)"
        )
    if tactic.family == "decoy" and tactic.decoy is None:
        reasons.append("decoy family requires a decoy")
    if tactic.family == "comms_cut" and tactic.comms_event is None:
        reasons.append("comms_cut family requires a comms event")
    return reasons


def strong_grid(site: Site, limits: Limits, n_phases: int = 48) -> list[Tactic]:
    """Every entry, straight to the asset, slowest and fastest, at n_phases even phases."""
    if n_phases < 1:
        raise ValueError(f"n_phases must be at least 1, got {n_phases}")
    speeds = sorted({limits.speed_min_mps, limits.speed_cap_mps})
    tactics: list[Tactic] = []
    for entry in site.entry_points:
        for speed in speeds:
            for k in range(n_phases):
                phase = k / n_phases
                tactic = Tactic(
                    id=f"strong-{entry.id}-{speed:g}-{phase:.4f}",
                    family=STRONG_FAMILY,
                    entry_id=entry.id,
                    phase=phase,
                    speed_mps=speed,
                    waypoints=[site.asset],
                    origin=STRONG_ORIGIN,
                )
                if t_reach_s(site, tactic) <= MAX_T_REACH_S:
                    tactics.append(tactic)
    return tactics


def key_hash(key: Sequence[int]) -> str:
    """First 8 hex of the sha256 of the key's JSON: two different keys never share ids."""
    return hashlib.sha256(json.dumps([int(k) for k in key]).encode()).hexdigest()[:8]


def _draw_inside(rng: np.random.Generator, perimeter: Sequence[XY]) -> XY:
    xmin, xmax = min(p.x for p in perimeter), max(p.x for p in perimeter)
    ymin, ymax = min(p.y for p in perimeter), max(p.y for p in perimeter)
    for _ in range(MAX_POINT_DRAWS):
        p = XY(
            x=round(float(rng.uniform(xmin, xmax)), COORD_DECIMALS),
            y=round(float(rng.uniform(ymin, ymax)), COORD_DECIMALS),
        )
        if _point_in_polygon(p, perimeter):
            return p
    raise ValueError("could not draw a point inside the perimeter: is the polygon degenerate?")


def _draw_speed(rng: np.random.Generator, limits: Limits) -> float:
    u = float(rng.random())
    if u < SNAP_PROBABILITY:
        return limits.speed_min_mps
    if u < 2 * SNAP_PROBABILITY:
        return limits.speed_cap_mps
    return float(rng.uniform(limits.speed_min_mps, limits.speed_cap_mps))


def random_tactics(site: Site, limits: Limits, n: int, key: Sequence[int]) -> list[Tactic]:
    """n seeded multi-waypoint tactics inside the limits; the same key gives the same list."""
    rng = np.random.default_rng([int(k) for k in key])
    prefix = f"rand-{key_hash(key)}"
    entries = list(site.entry_points)
    tactics: list[Tactic] = []
    draws = 0
    while len(tactics) < n:
        draws += 1
        if draws > MAX_TACTIC_DRAWS_PER_TACTIC * max(n, 1):
            raise ValueError(
                f"only {len(tactics)} of {n} random tactics fit the limits after {draws - 1} draws"
            )
        entry = entries[int(rng.integers(len(entries)))]
        phase = float(rng.random())
        speed = _draw_speed(rng, limits)
        n_mid = int(rng.integers(limits.max_waypoints))
        mids = [_draw_inside(rng, site.perimeter) for _ in range(n_mid)]
        tactic = Tactic(
            id=f"{prefix}-{len(tactics):04d}",
            family=RANDOM_FAMILY,
            entry_id=entry.id,
            phase=phase,
            speed_mps=speed,
            waypoints=[*mids, site.asset],
            origin=RANDOM_ORIGIN,
        )
        if not validate_tactic(site, limits, tactic):
            tactics.append(tactic)
    return tactics


def _candidates(node: Any) -> Iterator[dict[str, Any]]:
    """Every JSON object that looks like a tactic, however deeply it is wrapped."""
    if isinstance(node, dict):
        if all(marker in node for marker in TACTIC_MARKERS):
            yield node
        else:
            for value in node.values():
                yield from _candidates(value)
    elif isinstance(node, list):
        for value in node:
            yield from _candidates(value)


def _json_files(paths: Sequence[Path], notes: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.json") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            notes.append(f"{path}: no such file or directory")
    return files


def load_tactic_files(
    paths: Sequence[Path], site: Site, limits: Limits
) -> tuple[list[Tactic], list[str]]:
    """The valid tactics in the files, and a note for everything rejected. Never raises."""
    notes: list[str] = []
    found: dict[str, Tactic] = {}
    for file in _json_files(paths, notes):
        try:
            raw = json.loads(file.read_text())
        except (OSError, ValueError) as e:
            notes.append(f"{file}: unreadable, skipped ({e})")
            continue
        candidates = list(_candidates(raw))
        if not candidates:
            notes.append(f"{file}: holds no tactic, skipped")
            continue
        for index, candidate in enumerate(candidates):
            label = f"{file}[{index}] id={candidate.get('id')!r}"
            try:
                tactic = Tactic.model_validate(candidate)
            except ValidationError as e:
                first = e.errors()[0]
                where = ".".join(str(part) for part in first["loc"])
                notes.append(
                    f"{label}: fails the Tactic contract ({e.error_count()} errors, first: "
                    f"{where}: {first['msg']})"
                )
                continue
            reasons = validate_tactic(site, limits, tactic)
            if reasons:
                notes.append(f"{label}: outside the limits: {'; '.join(reasons)}")
                continue
            digest = tactic.content_hash()
            if digest in found:
                notes.append(f"{label}: exact duplicate of a tactic already loaded, dropped")
                continue
            found[digest] = tactic
    id_counts: dict[str, int] = {}
    for tactic in found.values():
        id_counts[tactic.id] = id_counts.get(tactic.id, 0) + 1
    tactics: list[Tactic] = []
    for digest, tactic in found.items():
        if id_counts[tactic.id] > 1:
            renamed = f"{tactic.id}-{digest[:6]}"
            notes.append(
                f"id {tactic.id!r} names {id_counts[tactic.id]} tactics: one is now {renamed!r}"
            )
            tactic = tactic.model_copy(update={"id": renamed})
        tactics.append(tactic)
    return tactics, notes


def keep_most_harmful(
    tactics: Sequence[Tactic], detection_by_id: Mapping[str, float], keep: int = 30
) -> list[Tactic]:
    """The keep tactics the defender detects least often, ties broken by id."""
    missing = sorted(t.id for t in tactics if t.id not in detection_by_id)
    if missing:
        raise ValueError(f"no measured detection for {len(missing)} tactics, first: {missing[:5]}")
    bad = sorted(t.id for t in tactics if math.isnan(detection_by_id[t.id]))
    if bad:
        raise ValueError(f"detection is NaN for {len(bad)} tactics, first: {bad[:5]}")
    ranked = sorted(tactics, key=lambda t: (detection_by_id[t.id], t.id))
    return ranked[: max(keep, 0)]
