"""Self-contained HTML clips of a miss and a catch, dropped in pitch/ for lane C."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from airtight.contracts.episode import (
    OutcomeEvent,
    PositionEvent,
    read_episode_log,
    write_episode_log,
)
from airtight.dimos_lane.replay import ReplayPlan, densify_intruder, plan_replay
from airtight.dimos_lane.site_io import load_example_site

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.contracts.site import XY, Site

SVG_W = 960
SVG_H = 640


def _scale(site: Site, x: float, y: float) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = site.bounds
    px = (x - xmin) / (xmax - xmin) * SVG_W
    py = SVG_H - (y - ymin) / (ymax - ymin) * SVG_H
    return px, py


def _poly(site: Site, pts: list[XY]) -> str:
    return " ".join(f"{_scale(site, p.x, p.y)[0]:.1f},{_scale(site, p.x, p.y)[1]:.1f}" for p in pts)


def render_html(plan: ReplayPlan, site: Site) -> str:
    perimeter = _poly(site, site.perimeter + [site.perimeter[0]])
    asset = _scale(site, site.asset.x, site.asset.y)
    frames: list[dict[str, object]] = []
    times = sorted(
        {t for t, _ in plan.intruder} | {t for pts in plan.markers.values() for t, _ in pts}
    )
    if not times:
        times = [0.0]
    for t in times:
        intruder = next((p for ts, p in reversed(plan.intruder) if ts <= t), None)
        agents = {}
        for oid, pts in plan.markers.items():
            pos = next((p for ts, p in reversed(pts) if ts <= t), None)
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
    title = f"{plan.title} seed={plan.seed} {plan.tactic_id}"
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
<div class="meta">timely_detected={plan.timely_detected} t_alarm={plan.t_alarm} t_cdp={plan.t_cdp} dispatch={plan.dispatch_result}</div>
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


def write_clip(plan: ReplayPlan, dest: Path, site: Site | None = None) -> Path:
    site = site or load_example_site()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_html(plan, site))
    sidecar = dest.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "title": plan.title,
                "seed": plan.seed,
                "tactic_id": plan.tactic_id,
                "timely_detected": plan.timely_detected,
                "t_alarm": plan.t_alarm,
                "dispatch_result": plan.dispatch_result,
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
