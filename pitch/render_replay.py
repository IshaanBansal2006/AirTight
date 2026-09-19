"""Render one episode log to an MP4: the yard, the fleet, the intruder, detections, the alarm and the deadline."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from palette import LIGHT, Palette

from airtight.contracts import (
    AlarmEvent,
    DetectionEvent,
    OutcomeEvent,
    PositionEvent,
    Site,
    read_episode_log,
)

REPO = Path(__file__).resolve().parents[1]


def load(log_path: Path):  # type: ignore[no-untyped-def]
    header, events = read_episode_log(log_path)
    positions: dict[float, dict[str, tuple[float, float]]] = defaultdict(dict)
    detections: list[tuple[float, str, str]] = []
    alarm_t: float | None = None
    outcome: OutcomeEvent | None = None
    for ev in events:
        if isinstance(ev, PositionEvent):
            positions[round(ev.t, 3)][ev.object_id] = (ev.position.x, ev.position.y)
        elif isinstance(ev, DetectionEvent):
            detections.append((ev.t, ev.agent_id, ev.object_id))
        elif isinstance(ev, AlarmEvent) and alarm_t is None:
            alarm_t = ev.t
        elif isinstance(ev, OutcomeEvent):
            outcome = ev
    if outcome is None:
        raise ValueError(f"{log_path} has no outcome event")
    if not positions:
        raise ValueError(
            f"{log_path} has no position events; render needs a full log, not a light one"
        )
    return header, positions, detections, alarm_t, outcome


def kind_of(object_id: str) -> str:
    if object_id == "intruder":
        return "intruder"
    if object_id.startswith("decoy"):
        return "decoy"
    if object_id.startswith(("drone", "go2", "guard")):
        return "agent"
    return "benign"


def render(
    log_path: Path, site: Site, out: Path, fps: int = 10, speed: float = 4.0, p: Palette = LIGHT
) -> Path:
    header, positions, detections, alarm_t, outcome = load(log_path)
    times = sorted(positions)
    dt = times[1] - times[0] if len(times) > 1 else 0.5
    stride = max(1, int(round(speed / (fps * dt))))
    frames = times[::stride]
    xmin, ymin, xmax, ymax = site.bounds
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.patch.set_facecolor(p.surface)
    ax.set_facecolor(p.surface)
    ax.add_patch(
        Polygon(
            [(q.x, q.y) for q in site.perimeter],
            closed=True,
            fill=False,
            edgecolor=p.ink,
            linewidth=1.5,
        )
    )
    for e in site.entry_points:
        ax.scatter([e.position.x], [e.position.y], marker="s", s=40, color=p.ink)
        ax.annotate(
            e.id,
            (e.position.x, e.position.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=p.ink_secondary,
        )
    for d in site.docks:
        ax.scatter([d.position.x], [d.position.y], marker="D", s=30, color=p.muted)
    ax.scatter(
        [site.asset.x],
        [site.asset.y],
        marker="*",
        s=200,
        color=p.critical,
        edgecolor=p.surface,
        zorder=5,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    colors = {"intruder": p.critical, "decoy": p.series[1], "agent": p.series[0], "benign": p.muted}
    sizes = {"intruder": 70, "decoy": 60, "agent": 55, "benign": 30}
    scat = {
        k: ax.scatter(
            [], [], s=sizes[k], color=c, edgecolor=p.surface, linewidth=1.0, zorder=4, label=k
        )
        for k, c in colors.items()
    }
    rings = ax.scatter(
        [], [], s=260, facecolors="none", edgecolors=p.series[1], linewidths=1.5, zorder=3
    )
    title = ax.set_title("", loc="left", fontsize=10, color=p.ink)
    banner = ax.text(
        0.5,
        0.04,
        "",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color=p.critical,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    verdict = "TIMELY DETECTION" if outcome.timely_detected else "MISSED"

    def update(t: float):  # type: ignore[no-untyped-def]
        pos = positions[t]
        for k, sc in scat.items():
            pts = [xy for oid, xy in pos.items() if kind_of(oid) == k]
            sc.set_offsets(pts if pts else [[-1e6, -1e6]])
        recent = [pos[obj] for (td, _, obj) in detections if t - 1.0 <= td <= t and obj in pos]
        rings.set_offsets(recent if recent else [[-1e6, -1e6]])
        title.set_text(
            f"{header.tactic.family} via {header.tactic.entry_id}  seed {header.seed}   t = {t:5.1f} s   deadline t_cdp = {outcome.t_cdp:.0f} s"
        )
        if alarm_t is not None and t >= alarm_t:
            banner.set_text(f"ALARM at {alarm_t:.0f} s: {verdict}")
        elif t >= outcome.t_cdp and outcome.t_cdp > 0:
            banner.set_text("deadline passed, no alarm")
        else:
            banner.set_text("")
        return [*scat.values(), rings, title, banner]

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument(
        "--site", type=Path, default=REPO / "scenarios" / "logistics_yard" / "site.json"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--speed", type=float, default=4.0, help="simulated seconds per real second")
    args = ap.parse_args(argv)
    site = Site.model_validate_json(args.site.read_text())
    out = args.out or (REPO / "pitch" / "clips" / (args.log.stem + ".mp4"))
    render(args.log, site, out, fps=args.fps, speed=args.speed)
    print(f"rendered {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
