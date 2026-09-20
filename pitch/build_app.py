"""Assemble the operator console: five pages built from the frozen report, the recorded episodes, the red-team rounds and the site model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from airtight.contracts import (
    AlarmEvent,
    DetectionEvent,
    FleetConfig,
    OutcomeEvent,
    PositionEvent,
    Report,
    ScoreEvent,
    SensorCurves,
    Site,
    read_episode_log,
)
from airtight.redteam.search import SearchResult

REPO = Path(__file__).resolve().parents[1]
SCEN = REPO / "scenarios" / "logistics_yard"


def episode(log_path: Path, fleet_name: str) -> dict:
    header, events = read_episode_log(log_path)
    tracks: dict[str, list[list[float]]] = {}
    detections: list[list] = []
    scores: list[list[float]] = []
    alarm_t = None
    via = None
    outcome = None
    first_det = None
    for ev in events:
        if isinstance(ev, PositionEvent):
            tracks.setdefault(ev.object_id, []).append(
                [round(ev.t, 2), round(ev.position.x, 2), round(ev.position.y, 2)]
            )
        elif isinstance(ev, DetectionEvent):
            detections.append([round(ev.t, 2), ev.agent_id, ev.object_id, bool(ev.true_positive)])
            if ev.object_id == "intruder" and first_det is None:
                first_det = ev.t
        elif isinstance(ev, ScoreEvent):
            if ev.object_id == "intruder":
                scores.append([round(ev.t, 2), round(ev.value, 3)])
        elif isinstance(ev, AlarmEvent) and alarm_t is None:
            alarm_t, via = ev.t, ev.via
        elif isinstance(ev, OutcomeEvent):
            outcome = ev
    if outcome is None:
        raise ValueError(f"{log_path} has no outcome")
    responder = next(
        (a for a in tracks if a.startswith("go2")),
        next((a for a in tracks if a.startswith("guard")), via or "drone_1"),
    )
    return {
        "fleet": fleet_name,
        "tactic": header.tactic.id,
        "family": header.tactic.family,
        "entry": header.tactic.entry_id,
        "phase": header.tactic.phase,
        "seed": header.seed,
        "tracks": tracks,
        "detections": detections,
        "scores": scores,
        "t_alarm": alarm_t,
        "via": via,
        "t_cdp": outcome.t_cdp,
        "t_end": outcome.t,
        "timely": outcome.timely_detected,
        "first_detection": first_det,
        "responder": responder,
    }


def priority_ramp() -> dict[str, list[str]]:
    """One hue, ordered by lightness: the chart palette's amber mixed toward paper and toward ink. Dark mode runs the other way."""
    sys.path.insert(0, str(REPO / "pitch"))
    from palette import LIGHT

    amber = [int(LIGHT.series[3][i : i + 2], 16) for i in (1, 3, 5)]

    def mix(to: int, k: float) -> str:
        return "#" + "".join(f"{round(c + (to - c) * k):02x}" for c in amber)

    light = [mix(255, 0.8), mix(255, 0.5), mix(255, 0.2), mix(0, 0.1), mix(0, 0.3), mix(0, 0.5)]
    return {"light": light, "dark": light[::-1]}


def perception_view(
    log_path: Path, fleet_name: str, fleets_dir: Path, site: Site, curves: SensorCurves
) -> dict | None:
    """Controller trace and dimOS perceived map for one recording, or None when either cannot be built."""
    sys.path.insert(0, str(REPO / "pitch"))
    from perception_map import build_view

    fleet_path = fleets_dir / f"{fleet_name}.json"
    if not fleet_path.exists():
        return None
    return build_view(
        log_path, FleetConfig.model_validate_json(fleet_path.read_text()), site, curves
    )


def rounds(minmax_dir: Path, site: Site) -> list[dict]:
    """Every attack-fix-re-attack round with the worst tactic's path, when the run is on disk."""
    summary = minmax_dir / "summary.json"
    if not summary.exists():
        return []
    entries = {e.id: e.position for e in site.entry_points}
    out = []
    for row in json.loads(summary.read_text()):
        k = int(row["k"])
        top = minmax_dir / f"iter{k}_tactics" / f"top_{row['worst_family']}.json"
        tactic = None
        if top.exists():
            res = SearchResult.model_validate_json(top.read_text())
            t, s = res.best()
            entry = entries.get(t.entry_id)
            path = ([[entry.x, entry.y]] if entry else []) + [[w.x, w.y] for w in t.waypoints]
            tactic = {
                "id": t.id,
                "entry": t.entry_id,
                "phase": t.phase,
                "speed": t.speed_mps,
                "path": [[round(x, 1), round(y, 1)] for x, y in path],
                "decoy": [t.decoy.position.x, t.decoy.position.y] if t.decoy else None,
                "miss": s.miss_rate,
                "origin": t.origin,
            }
        out.append({**row, "tactic": tactic})
    return out


def archived_runs() -> list[dict]:
    runs = []
    for m in sorted((REPO / "results").glob("*/manifest.json")):
        try:
            d = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        runs.append(
            {
                "name": d.get("name", m.parent.name),
                "archived_at": d.get("archived_at"),
                "git_commit": d.get("git_commit"),
                "n_files": len(d.get("files", [])),
                "path": f"results/{m.parent.name}/",
                "reproductions": d.get("reproductions", []),
            }
        )
    return runs


def threats(tactics_dir: Path) -> list[dict]:
    out = []
    for f in sorted(tactics_dir.glob("top_*.json")):
        res = SearchResult.model_validate_json(f.read_text())
        t, s = res.best()
        out.append(
            {
                "family": t.family,
                "entry": t.entry_id,
                "phase": t.phase,
                "speed": t.speed_mps,
                "origin": t.origin,
                "miss": s.miss_rate,
                "id": t.id,
            }
        )
    out.sort(key=lambda x: -x["miss"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=REPO / "pitch" / "report.json")
    ap.add_argument("--charts", type=Path, default=REPO / "pitch" / "charts")
    ap.add_argument("--tactics-dir", type=Path, default=REPO / "data" / "v3" / "tactics")
    ap.add_argument("--clips", type=Path, default=REPO / "pitch" / "clips" / "clips_c.json")
    ap.add_argument("--minmax-dir", type=Path, default=REPO / "results" / "minmax")
    ap.add_argument("--fleets-dir", type=Path, default=SCEN / "fleets")
    ap.add_argument("--template", type=Path, default=REPO / "pitch" / "app_template.html")
    ap.add_argument("--clip-logs", type=Path, default=REPO / "data" / "clip_logs")
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "app.html")
    args = ap.parse_args(argv)

    report = Report.model_validate_json(args.report.read_text())
    site = Site.model_validate_json((SCEN / "site.json").read_text())
    curves = SensorCurves.model_validate_json((SCEN / "sensor_curve.json").read_text())
    numbers = json.loads((args.charts / "numbers.json").read_text())
    clips = json.loads(args.clips.read_text())
    fixed_name = numbers.get("before_after", {}).get("fixed") or report.baseline_config
    fleet_path = args.fleets_dir / f"{fixed_name}.json"
    fleet = FleetConfig.model_validate_json(fleet_path.read_text()) if fleet_path.exists() else None
    cond = report.conditions
    tokens_path = args.charts / "token_numbers.json"
    tokens = json.loads(tokens_path.read_text()) if tokens_path.exists() else None
    miss_log = (
        args.clip_logs
        / "miss"
        / f"{clips['baseline']}__{clips['tactic_id']}__{clips['seed']}.jsonl"
    )
    catch_log = (
        args.clip_logs / "catch" / f"{clips['fixed']}__{clips['tactic_id']}__{clips['seed']}.jsonl"
    )
    payload = {
        "built_at": datetime.now(UTC).isoformat(timespec="minutes"),
        "site": site.model_dump(mode="json"),
        "curves": {
            k: {"fov_deg": c.fov_deg, "range": c.max_range_m()} for k, c in curves.curves.items()
        },
        "fleet": None
        if fleet is None
        else {
            "name": fleet.name,
            "agents": [a.model_dump(mode="json") for a in fleet.agents],
            "stagger": fleet.charge_policy.stagger_offsets_s,
            "cost_by_type": fleet.cost_per_hour_by_type,
            "cost": fleet.cost_per_hour(),
        },
        "report": {
            "baseline": report.baseline_config,
            "fixed": fixed_name,
            "n_seeds": cond.n_seeds,
            "far": cond.far_per_hour_operating_point,
            "conditions": f"Operating point {cond.far_per_hour_operating_point:g} false alarm/h, {cond.n_seeds} seeds, {cond.sensor_calibration}, {cond.detection_model_note}",
            "configs": [
                {
                    "name": c.config_name,
                    "cost": c.cost_per_hour,
                    "pd": c.pd_at_operating_point,
                    "ci": list(c.pd_at_operating_point_ci),
                    "worst_pd": c.worst_tactic_pd,
                    "worst_id": c.worst_tactic_id,
                    "blind_pd": c.worst_tactic_pd_schedule_blind,
                    "decisions": c.human_decisions_per_hour,
                    "gap": c.coverage_gap_s_per_hour,
                }
                for c in sorted(report.configs, key=lambda c: c.cost_per_hour)
            ],
        },
        "threats": threats(args.tactics_dir),
        "episodes": {
            "miss": episode(miss_log, clips["baseline"]),
            "catch": episode(catch_log, clips["fixed"]),
        },
        "rounds": rounds(args.minmax_dir, site),
        "tokens": tokens,
        "runs": archived_runs(),
        "ramp": priority_ramp(),
    }
    for kind, name, log in (
        ("miss", clips["baseline"], miss_log),
        ("catch", clips["fixed"], catch_log),
    ):
        payload["episodes"][kind]["view"] = perception_view(
            log, name, args.fleets_dir, site, curves
        )
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = args.template.read_text().replace("__DATA__", blob)
    args.out.write_text(html)
    print(
        f"app written to {args.out} ({len(html) / 1e6:.2f} MB), rounds: {len(payload['rounds'])}, runs: {len(payload['runs'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
