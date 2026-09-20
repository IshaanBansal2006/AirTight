"""Self-contained HTML clips of a miss and a catch, dropped in pitch/ for lane C."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from airtight.contracts import XY, Tactic
from airtight.contracts.episode import (
    OutcomeEvent,
    PositionEvent,
    read_episode_log,
    write_episode_log,
)
from airtight.dimos_lane.replay import ReplayPlan, densify_intruder, plan_replay, pose_at_or_before
from airtight.dimos_lane.site_io import (
    load_example_site,
    load_logistics_curves,
    load_logistics_fleet,
    load_logistics_site,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from airtight.contracts.episode import EpisodeResult
    from airtight.contracts.fleet import FleetConfig
    from airtight.contracts.sensor_curve import SensorCurves
    from airtight.contracts.site import Site

SVG_W = 960
SVG_H = 640

REPO = Path(__file__).resolve().parents[3]
HANDOFF_FAMILY = "charging_window"
HANDOFF_ENTRY = "rear_fence_gap"
HANDOFF_BASELINE = "d2_go2_guard_sync"
HANDOFF_FIXED = "d3_go2_guard_stagger"
HANDOFF_MAX_SEEDS = 20
CLIPS_JSON_KEYS = (
    "tactic_id",
    "family",
    "entry",
    "phase",
    "seed",
    "baseline",
    "fixed",
    "miss_t_alarm",
    "catch_t_alarm",
    "t_cdp",
)


def _scale(site: Site, x: float, y: float) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = site.bounds
    px = (x - xmin) / (xmax - xmin) * SVG_W
    py = SVG_H - (y - ymin) / (ymax - ymin) * SVG_H
    return px, py


def _poly(site: Site, pts: list[XY]) -> str:
    return " ".join(f"{_scale(site, p.x, p.y)[0]:.1f},{_scale(site, p.x, p.y)[1]:.1f}" for p in pts)


def render_html(plan: ReplayPlan, site: Site, *, clean: bool = False) -> str:
    perimeter = _poly(site, site.perimeter + [site.perimeter[0]])
    asset = _scale(site, site.asset.x, site.asset.y)
    frames: list[dict[str, object]] = []
    times = sorted(
        {t for t, _ in plan.intruder} | {t for pts in plan.markers.values() for t, _ in pts}
    )
    if not times:
        times = [0.0]
    intruder_t = [ts for ts, _ in plan.intruder]
    marker_t = {oid: [ts for ts, _ in pts] for oid, pts in plan.markers.items()}
    for t in times:
        intruder = pose_at_or_before(plan.intruder, t, intruder_t)
        agents = {}
        for oid, pts in plan.markers.items():
            pos = pose_at_or_before(pts, t, marker_t[oid])
            if pos is not None:
                agents[oid] = {"x": pos.x, "y": pos.y}
        if intruder is not None:
            ix, iy = _scale(site, intruder.x, intruder.y)
        else:
            ix = iy = 0.0
        frames.append(
            {
                "t": t,
                "intruder": [ix, iy],
                "agents": {
                    oid: list(_scale(site, pos["x"], pos["y"])) for oid, pos in agents.items()
                },
                "alarm": plan.t_alarm is not None and t >= plan.t_alarm,
            }
        )
    if clean:
        which = "Replay A — miss" if not plan.timely_detected else "Replay B — catch"
        title = f"{which} · seed {plan.seed} · {plan.tactic_id}"
        alarm = "no timely alarm" if plan.t_alarm is None else f"alarm at {plan.t_alarm:.0f}s"
        meta = f"{alarm} · deadline t_cdp={plan.t_cdp:.0f}s"
    else:
        title = f"{plan.title} seed={plan.seed} {plan.tactic_id}"
        meta = (
            f"timely_detected={plan.timely_detected} t_alarm={plan.t_alarm} "
            f"t_cdp={plan.t_cdp} dispatch={plan.dispatch_result}"
        )
    payload = json.dumps({"frames": frames, "title": title, "timely": plan.timely_detected})
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; margin: 24px; }}
svg {{ background: #1b1f16; border: 1px solid #333; }}
.banner {{ font-size: 20px; margin-bottom: 8px; }}
.meta {{ color: #aaa; margin-bottom: 12px; }}
</style></head>
<body>
<div class="banner">{title}</div>
<div class="meta">{meta}</div>
<svg id="s" width="{SVG_W}" height="{SVG_H}">
  <polygon points="{perimeter}" fill="#24301c" stroke="#8f8" stroke-width="3"/>
  <circle cx="{asset[0]:.1f}" cy="{asset[1]:.1f}" r="8" fill="#c33"/>
  <circle id="intruder" r="7" fill="#fc0"/>
</svg>
<script>
const data = {payload};
const svg = document.getElementById('s');
const intr = document.getElementById('intruder');
const agents = {{}};
function dot(id, color) {{
  let el = agents[id];
  if (!el) {{
    el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    el.setAttribute('r', '6');
    el.setAttribute('fill', color);
    svg.appendChild(el);
    agents[id] = el;
  }}
  return el;
}}
let i = 0;
function tick() {{
  const f = data.frames[i];
  intr.setAttribute('cx', f.intruder[0]);
  intr.setAttribute('cy', f.intruder[1]);
  intr.setAttribute('fill', f.alarm ? '#f55' : '#fc0');
  for (const [id, xy] of Object.entries(f.agents)) {{
    const el = dot(id, id.includes('go2') ? '#6cf' : '#9af');
    el.setAttribute('cx', xy[0]);
    el.setAttribute('cy', xy[1]);
  }}
  i = (i + 1) % data.frames.length;
}}
setInterval(tick, 120);
tick();
</script>
</body></html>
"""


def write_clip(
    plan: ReplayPlan, dest: Path, site: Site | None = None, *, clean: bool = False
) -> Path:
    site = site or load_example_site()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_html(plan, site, clean=clean))
    sidecar = dest.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "title": plan.title,
                "seed": plan.seed,
                "tactic_id": plan.tactic_id,
                "timely_detected": plan.timely_detected,
                "t_alarm": plan.t_alarm,
                "dispatch_result": None if clean else plan.dispatch_result,
            },
            indent=2,
        )
        + "\n"
    )
    return dest


def miss_log_from_catch(src: Path, dest: Path) -> Path:
    """Clone a catch log into a miss of the same seed: drop the alarm, flip outcome."""
    header, event_iter = read_episode_log(src)
    kept = []
    outcome = None
    for event in event_iter:
        if getattr(event, "kind", None) == "alarm_delivered":
            continue
        if isinstance(event, OutcomeEvent):
            outcome = event.model_copy(
                update={"timely_detected": False, "t_alarm": None, "human_decisions": 0}
            )
            continue
        kept.append(event)
    if outcome is None:
        raise ValueError(f"{src} has no outcome event")
    # last known position still present
    if not any(isinstance(e, PositionEvent) for e in kept):
        raise ValueError(f"{src} has no positions to replay")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_episode_log(dest, header, [*kept, outcome])
    return dest


def write_demo_clips(example_log: Path, pitch_dir: Path) -> tuple[Path, Path]:
    site = load_example_site()
    catch_plan = densify_intruder(plan_replay(example_log))
    miss_log = miss_log_from_catch(example_log, pitch_dir / "logs" / "miss.jsonl")
    miss_plan = densify_intruder(plan_replay(miss_log))
    catch_html = write_clip(catch_plan, pitch_dir / "clips" / "catch.html", site)
    miss_html = write_clip(miss_plan, pitch_dir / "clips" / "miss.html", site)
    return miss_html, catch_html


def charging_window_tactic(
    site: Site,
    fleet: FleetConfig | None = None,
    curves: SensorCurves | None = None,
    tactics_dir: Path | None = None,
) -> Tactic:
    """Charging-window path through high-value cells the baseline fleet has not watched recently.

    If lane C dropped a searched `top_charging_window.json`, use that instead.
    """
    from airtight.dimos_lane.stale_path import charging_window_from_staleness

    root = tactics_dir or (REPO / "data" / "v3" / "tactics")
    top = root / "top_charging_window.json"
    if top.is_file():
        payload = json.loads(top.read_text())
        for raw in payload.get("tactics") or []:
            tactic = Tactic.model_validate(raw)
            if tactic.family == HANDOFF_FAMILY and tactic.entry_id == HANDOFF_ENTRY:
                return tactic
    used_fleet = fleet or load_logistics_fleet(HANDOFF_BASELINE)
    used_curves = curves or load_logistics_curves()
    return charging_window_from_staleness(site, used_fleet, used_curves, entry_id=HANDOFF_ENTRY)


def clips_record(
    tactic: Tactic,
    seed: int,
    miss: EpisodeResult,
    catch: EpisodeResult,
    *,
    baseline: str = HANDOFF_BASELINE,
    fixed: str = HANDOFF_FIXED,
) -> dict[str, object]:
    return {
        "tactic_id": tactic.id,
        "family": tactic.family,
        "entry": tactic.entry_id,
        "phase": tactic.phase,
        "seed": seed,
        "baseline": baseline,
        "fixed": fixed,
        "miss_t_alarm": miss.t_alarm,
        "catch_t_alarm": catch.t_alarm,
        "t_cdp": catch.t_cdp,
    }


def _seed_list(seeds_path: Path | None, max_seeds: int) -> list[int]:
    path = seeds_path or (REPO / "data" / "seeds.json")
    return [int(s) for s in json.loads(path.read_text())["seeds"][:max_seeds]]


def _render_mp4(log: Path, site: Site, dest: Path) -> Path | None:
    pitch = REPO / "pitch"
    if str(pitch) not in sys.path:
        sys.path.insert(0, str(pitch))
    try:
        from render_replay import render  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        render(log, site, dest, fps=12, speed=3.0)
    except Exception:
        return None
    return dest if dest.is_file() else None


def find_miss_catch_pair(
    site: Site,
    baseline: FleetConfig,
    fixed: FleetConfig,
    tactic: Tactic,
    curves: SensorCurves,
    seeds: Sequence[int],
    log_dir: Path,
    run_episode: Callable[..., EpisodeResult],
) -> tuple[int, EpisodeResult, EpisodeResult]:
    for seed in seeds:
        miss = run_episode(site, baseline, tactic, curves, seed, log_dir / "miss")
        catch = run_episode(site, fixed, tactic, curves, seed, log_dir / "catch")
        if not miss.timely_detected and catch.timely_detected:
            return seed, miss, catch
    raise RuntimeError(
        f"no seed in the first {len(seeds)} where {baseline.name} misses and "
        f"{fixed.name} catches {tactic.id}"
    )


def write_handoff_clips(
    pitch_dir: Path,
    *,
    seeds: Sequence[int] | None = None,
    seeds_path: Path | None = None,
    max_seeds: int = HANDOFF_MAX_SEEDS,
    log_dir: Path | None = None,
    tactics_dir: Path | None = None,
    render_mp4: bool = True,
    run_episode: Callable[..., EpisodeResult] | None = None,
) -> Path:
    """Same-seed miss/catch on logistics_yard for C: HTML + clips.json (+ MP4 if ffmpeg)."""
    if run_episode is None:
        from airtight.sim.runner import run_episode as _run_episode

        run_episode = _run_episode
    os.environ.setdefault("AIRTIGHT_ENGINE", "v0")
    site = load_logistics_site()
    curves = load_logistics_curves()
    baseline = load_logistics_fleet(HANDOFF_BASELINE)
    fixed = load_logistics_fleet(HANDOFF_FIXED)
    tactic = charging_window_tactic(site, fleet=baseline, curves=curves, tactics_dir=tactics_dir)
    seed_ids = list(seeds) if seeds is not None else _seed_list(seeds_path, max_seeds)
    out = pitch_dir / "clips"
    logs = log_dir or (REPO / "data" / "clip_logs")
    seed, miss, catch = find_miss_catch_pair(
        site, baseline, fixed, tactic, curves, seed_ids, logs, run_episode
    )
    miss_plan = densify_intruder(plan_replay(miss.log_path, dispatch=False))
    catch_plan = densify_intruder(plan_replay(catch.log_path, dispatch=False))
    write_clip(miss_plan, out / "miss.html", site, clean=True)
    write_clip(catch_plan, out / "catch.html", site, clean=True)
    if render_mp4:
        _render_mp4(miss.log_path, site, out / "miss.mp4")
        _render_mp4(catch.log_path, site, out / "catch.mp4")
    manifest = out / "clips.json"
    manifest.write_text(json.dumps(clips_record(tactic, seed, miss, catch), indent=2) + "\n")
    return manifest
