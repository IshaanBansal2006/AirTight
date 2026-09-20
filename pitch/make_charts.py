"""Every number on a slide comes from here. Reads report.json, site.json and the search outputs; writes PNGs and numbers.json."""

from __future__ import annotations

import argparse
import json
import textwrap
from importlib import resources
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
from palette import DARK, FAMILY_ORDER, LIGHT, Palette

from airtight.contracts import ConfigResult, Report, SensorCurves, Site, Tactic
from airtight.redteam.coverage import GeometryCoverage
from airtight.redteam.geometry import point_in_polygon
from airtight.redteam.search import SearchResult

EXAMPLES = resources.files("airtight.contracts.examples")
REPO = Path(__file__).resolve().parents[1]


def style(p: Palette) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": p.surface,
            "axes.facecolor": p.surface,
            "axes.edgecolor": p.axis,
            "axes.labelcolor": p.ink_secondary,
            "axes.titlecolor": p.ink,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": p.grid,
            "grid.linewidth": 0.6,
            "xtick.color": p.muted,
            "ytick.color": p.muted,
            "text.color": p.ink,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "savefig.dpi": 200,
            "savefig.facecolor": p.surface,
        }
    )


def conditions_line(r: Report, width: int = 150) -> str:
    c = r.conditions
    text = (
        f"Operating point {c.far_per_hour_operating_point:g} false alarm/h · {c.n_seeds} seeds · "
        f"adversary: {c.adversary_knowledge} · sensor: {c.sensor_calibration} · {c.detection_model_note}"
    )
    return "\n".join(textwrap.wrap(text, width))


def _frontier(cfgs: list[ConfigResult]) -> list[ConfigResult]:
    best: list[ConfigResult] = []
    top = -1.0
    for c in sorted(cfgs, key=lambda c: (c.cost_per_hour, -c.pd_at_operating_point)):
        if c.pd_at_operating_point > top:
            best.append(c)
            top = c.pd_at_operating_point
    return best


def chart_cost_vs_detection(r: Report, p: Palette, out: Path, numbers: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    front = _frontier(r.configs)
    ax.plot(
        [c.cost_per_hour for c in front],
        [c.pd_at_operating_point for c in front],
        color=p.axis,
        linewidth=1.2,
        zorder=1,
    )
    placed: list[tuple[float, float]] = []
    pts = [(c.cost_per_hour, c.pd_at_operating_point) for c in r.configs]
    xspan = (max(x for x, _ in pts) - min(x for x, _ in pts)) or 1.0
    for c in sorted(r.configs, key=lambda c: (c.cost_per_hour, -c.pd_at_operating_point)):
        base = c.config_name == r.baseline_config
        color = p.series[1] if base else p.series[0]
        lo, hi = c.pd_at_operating_point_ci
        x, y = c.cost_per_hour, c.pd_at_operating_point
        ax.plot([x, x], [lo, hi], color=color, linewidth=1.2, alpha=0.7, zorder=2)
        ax.scatter([x], [y], s=64, color=color, edgecolor=p.surface, linewidth=1.5, zorder=3)
        xmin_pts = min(px for px, _ in pts)
        near_left_edge = (x - xmin_pts) / xspan < 0.15
        crowded_right = (not near_left_edge) and any(
            0 < (ox - x) / xspan < 0.3 and abs(oy - y) < 0.1 for ox, oy in pts
        )
        dy = 6.0
        row_conflict = False
        for px, py in placed:
            same_column = abs(px - x) / xspan < 0.12 and abs(py - y) < 0.06
            same_row = 0 < (x - px) / xspan < 0.4 and abs(py - y) < 0.05
            row_conflict = row_conflict or same_row
            if same_column or same_row:
                dy -= 11.0
        below = crowded_right and row_conflict
        placed.append((x, y))
        ax.annotate(
            c.config_name + (" (baseline)" if base else ""),
            (x, y),
            xytext=(0, -14) if below else (-6 if crowded_right else 6, dy),
            textcoords="offset points",
            ha="center" if below else ("right" if crowded_right else "left"),
            fontsize=7,
            color=p.ink_secondary,
        )
    ax.set_xlabel("Fleet cost, USD per hour")
    ax.set_ylabel("Timely detection probability at the operating point")
    ax.set_ylim(0, 1.02)
    ax.set_title("Detection versus cost, worst-case tactic per configuration")
    fig.text(0.01, 0.01, conditions_line(r), fontsize=7, color=p.muted)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "cost_vs_detection.png")
    plt.close(fig)
    numbers["cost_vs_detection"] = {
        c.config_name: {
            "cost_per_hour": c.cost_per_hour,
            "pd": c.pd_at_operating_point,
            "ci": list(c.pd_at_operating_point_ci),
        }
        for c in r.configs
    }
    numbers["frontier"] = [c.config_name for c in front]


def chart_roc(r: Report, p: Palette, out: Path, numbers: dict) -> None:
    ranked = sorted(r.configs, key=lambda c: c.pd_at_operating_point, reverse=True)
    picks: list[ConfigResult] = [r.config(r.baseline_config)] + [
        c for c in ranked if c.config_name != r.baseline_config
    ][:3]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, c in enumerate(picks):
        pts = sorted(c.roc, key=lambda q: q.far_per_hour)
        xs, ys = [q.far_per_hour for q in pts], [q.pd for q in pts]
        ax.plot(xs, ys, color=p.series[i], label=c.config_name)
        ax.fill_between(
            xs,
            [q.pd_ci[0] for q in pts],
            [q.pd_ci[1] for q in pts],
            color=p.series[i],
            alpha=0.12,
            linewidth=0,
        )
        ax.annotate(
            c.config_name,
            (xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=p.series[i],
            va="center",
        )
    ax.axvline(
        r.conditions.far_per_hour_operating_point, color=p.axis, linewidth=1, linestyle=(0, (3, 3))
    )
    ax.set_xlabel("False alarms per benign hour")
    ax.set_ylabel("Timely detection probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Detection against false-alarm rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, conditions_line(r), fontsize=7, color=p.muted)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "roc.png")
    plt.close(fig)
    numbers["roc_configs"] = [c.config_name for c in picks]


def _load_worst(tactics_dir: Path, r: Report) -> dict[str, Tactic]:
    worst: dict[str, Tactic] = {}
    wanted = {c.worst_tactic_id for c in r.configs}
    for fam in FAMILY_ORDER:
        f = tactics_dir / f"top_{fam}.json"
        if f.exists():
            res = SearchResult.model_validate_json(f.read_text())
            for t in res.tactics:
                if t.id in wanted or fam not in worst:
                    worst[fam] = t
                    if t.id in wanted:
                        break
    return worst


def chart_vulnerability_map(
    r: Report,
    site: Site,
    curves: SensorCurves,
    tactics_dir: Path,
    p: Palette,
    out: Path,
    numbers: dict,
) -> None:
    cm = GeometryCoverage().coverage(site, curves)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    xmin, ymin, xmax, ymax = site.bounds
    top = max(max(row) for row in cm.scores) or 1.0
    for iy in range(cm.ny):
        for ix in range(cm.nx):
            s = cm.scores[iy][ix]
            if s > 0 and point_in_polygon(cm.center(ix, iy), site.perimeter):
                step = min(len(p.sequential) - 1, int(round(s / top * (len(p.sequential) - 1))))
                ax.add_patch(
                    plt.Rectangle(
                        (cm.x0 + ix * cm.cell_m, cm.y0 + iy * cm.cell_m),
                        cm.cell_m,
                        cm.cell_m,
                        color=p.sequential[step],
                        linewidth=0,
                        alpha=0.8,
                        zorder=0,
                    )
                )
    ax.add_patch(
        Polygon(
            [(q.x, q.y) for q in site.perimeter],
            closed=True,
            fill=False,
            edgecolor=p.ink,
            linewidth=1.5,
            zorder=2,
        )
    )
    for s in site.fixed_sensors:
        c = curves.curves[s.sensor_type]
        ax.add_patch(
            Wedge(
                (s.position.x, s.position.y),
                c.max_range_m(),
                s.heading_deg - c.fov_deg / 2,
                s.heading_deg + c.fov_deg / 2,
                fill=False,
                edgecolor=p.muted,
                linewidth=0.8,
                linestyle=(0, (2, 2)),
                zorder=2,
            )
        )
    for e in site.entry_points:
        ax.scatter([e.position.x], [e.position.y], marker="s", s=50, color=p.ink, zorder=4)
        ax.annotate(
            e.id,
            (e.position.x, e.position.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=p.ink_secondary,
        )
    for d in site.docks:
        ax.scatter([d.position.x], [d.position.y], marker="D", s=40, color=p.muted, zorder=4)
        ax.annotate(
            d.id,
            (d.position.x, d.position.y),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=7,
            color=p.muted,
        )
    ax.scatter(
        [site.asset.x],
        [site.asset.y],
        marker="*",
        s=220,
        color=p.critical,
        edgecolor=p.surface,
        zorder=5,
    )
    ax.annotate(
        "asset",
        (site.asset.x, site.asset.y),
        xytext=(8, -4),
        textcoords="offset points",
        fontsize=8,
        color=p.ink,
    )
    worst = _load_worst(tactics_dir, r)
    for i, fam in enumerate(FAMILY_ORDER):
        t = worst.get(fam)
        if t is None:
            continue
        pts = [site.entry(t.entry_id).position, *t.waypoints]
        ax.plot(
            [q.x for q in pts],
            [q.y for q in pts],
            color=p.series[i],
            label=f"{fam} (phase {t.phase:.2f}, {t.speed_mps:.1f} m/s)",
            zorder=3,
        )
        if t.decoy is not None:
            ax.scatter(
                [t.decoy.position.x],
                [t.decoy.position.y],
                marker="x",
                s=70,
                color=p.series[i],
                zorder=4,
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(f"Vulnerability map: worst tactic per family on '{site.name}'")
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.01, 1.0))
    fig.text(
        0.01,
        0.01,
        "Shading: geometry coverage (fixed sensors and dock halos), darker is more watched. Dashed: fixed-sensor fields of view.",
        fontsize=7,
        color=p.muted,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "vulnerability_map.png")
    plt.close(fig)
    numbers["worst_tactics"] = {
        fam: {
            "id": t.id,
            "entry": t.entry_id,
            "phase": t.phase,
            "speed_mps": t.speed_mps,
            "origin": t.origin,
        }
        for fam, t in worst.items()
    }


def chart_before_after(r: Report, fixed: str | None, p: Palette, out: Path, numbers: dict) -> None:
    base = r.config(r.baseline_config)
    others = [c for c in r.configs if c.config_name != r.baseline_config]
    if not others:
        return
    after = (
        r.config(fixed)
        if fixed
        else max(others, key=lambda c: c.pd_at_operating_point - base.pd_at_operating_point)
    )
    metrics = [
        (
            "Timely detection\nat operating point",
            base.pd_at_operating_point,
            after.pd_at_operating_point,
            base.pd_at_operating_point_ci,
            after.pd_at_operating_point_ci,
            True,
        ),
        (
            "Detection against the\nre-attacking worst tactic",
            base.worst_tactic_pd,
            after.worst_tactic_pd,
            None,
            None,
            True,
        ),
        (
            "Human decisions\nper hour",
            base.human_decisions_per_hour,
            after.human_decisions_per_hour,
            None,
            None,
            False,
        ),
        (
            "Coverage gap,\nseconds per hour",
            base.coverage_gap_s_per_hour,
            after.coverage_gap_s_per_hour,
            None,
            None,
            False,
        ),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4.2))
    for ax, (name, b, a, bci, aci, higher_is_better) in zip(axes, metrics, strict=True):
        ax.plot([0, 1], [b, a], color=p.axis, linewidth=1.2, zorder=1)
        for x, v, ci, color in ((0, b, bci, p.muted), (1, a, aci, p.series[0])):
            if ci:
                ax.plot([x, x], list(ci), color=color, linewidth=1.2, alpha=0.7)
            ax.scatter([x], [v], s=70, color=color, edgecolor=p.surface, linewidth=1.5, zorder=3)
            ax.annotate(
                f"{v:.2f}" if v < 10 else f"{v:.0f}",
                (x, v),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=p.ink,
            )
        if a == b:
            verdict, color = "unchanged", p.muted
        elif (a > b) if higher_is_better else (a < b):
            verdict, color = "better", p.good
        else:
            verdict, color = "worse", p.critical
        ax.set_title(f"{name}\n{verdict}", fontsize=8, color=color)
        ax.set_xticks([0, 1], [base.config_name, after.config_name], fontsize=7)
        ax.set_xlim(-0.4, 1.4)
        ax.margins(y=0.3)
    fig.suptitle(
        f"Before and after the fix: {base.config_name} to {after.config_name} on the same seeds",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(0.01, 0.01, conditions_line(r, 200), fontsize=7, color=p.muted)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(out / "before_after.png")
    plt.close(fig)
    numbers["before_after"] = {
        "baseline": base.config_name,
        "fixed": after.config_name,
        "pd": [base.pd_at_operating_point, after.pd_at_operating_point],
        "worst_tactic_pd": [base.worst_tactic_pd, after.worst_tactic_pd],
        "human_decisions_per_hour": [base.human_decisions_per_hour, after.human_decisions_per_hour],
        "coverage_gap_s_per_hour": [base.coverage_gap_s_per_hour, after.coverage_gap_s_per_hour],
        "paired_deltas": [d.model_dump() for d in after.paired_vs_baseline],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render pitch charts from report.json; never type a number by hand."
    )
    ap.add_argument("--report", type=Path, default=REPO / "data" / "report.json")
    ap.add_argument("--site", type=Path, default=Path(str(EXAMPLES.joinpath("site.json"))))
    ap.add_argument(
        "--curves", type=Path, default=Path(str(EXAMPLES.joinpath("sensor_curve.json")))
    )
    ap.add_argument("--tactics-dir", type=Path, default=REPO / "data" / "tactics")
    ap.add_argument(
        "--fixed",
        default=None,
        help="config name shown as the 'after'; default is the largest detection gain over baseline",
    )
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "charts")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args(argv)
    report_path = (
        args.report if args.report.exists() else Path(str(EXAMPLES.joinpath("report.json")))
    )
    r = Report.model_validate_json(report_path.read_text())
    site = Site.model_validate_json(args.site.read_text())
    curves = SensorCurves.model_validate_json(args.curves.read_text())
    p = DARK if args.dark else LIGHT
    style(p)
    args.out.mkdir(parents=True, exist_ok=True)
    numbers: dict = {
        "report": str(report_path),
        "conditions": r.conditions.model_dump(),
        "n_configs": len(r.configs),
    }
    minmax_path = REPO / "data" / "minmax" / "summary.json"
    numbers["minmax"] = json.loads(minmax_path.read_text()) if minmax_path.exists() else []
    chart_cost_vs_detection(r, p, args.out, numbers)
    chart_roc(r, p, args.out, numbers)
    chart_vulnerability_map(r, site, curves, args.tactics_dir, p, args.out, numbers)
    chart_before_after(r, args.fixed, p, args.out, numbers)
    (args.out / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(
        f"charts and numbers.json written to {args.out} from {report_path.name}"
        + (" (EXAMPLE report, not a result)" if report_path.parent.name == "examples" else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
