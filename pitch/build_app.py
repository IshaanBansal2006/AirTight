"""Assemble the phone console (app.html) from the report, the site, the search output and two episode logs."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from airtight.contracts import (
    AlarmEvent,
    DetectionEvent,
    OutcomeEvent,
    PositionEvent,
    Report,
    SensorCurves,
    Site,
    read_episode_log,
)
from airtight.redteam.search import SearchResult

REPO = Path(__file__).resolve().parents[1]
SCEN = REPO / "scenarios" / "logistics_yard"


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def episode(log_path: Path, fleet_name: str) -> dict:
    header, events = read_episode_log(log_path)
    tracks: dict[str, list[list[float]]] = {}
    detections: list[list] = []
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
            detections.append([round(ev.t, 2), ev.agent_id, ev.object_id])
            if ev.object_id == "intruder" and first_det is None:
                first_det = ev.t
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
        "entry": header.tactic.entry_id,
        "seed": header.seed,
        "tracks": tracks,
        "detections": detections,
        "t_alarm": alarm_t,
        "t_cdp": outcome.t_cdp,
        "t_end": outcome.t,
        "timely": outcome.timely_detected,
        "first_detection": first_det,
        "responder": responder,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=REPO / "pitch" / "report.json")
    ap.add_argument("--charts", type=Path, default=REPO / "pitch" / "charts")
    ap.add_argument("--tactics-dir", type=Path, default=REPO / "data" / "v3" / "tactics")
    ap.add_argument("--clips", type=Path, default=REPO / "pitch" / "clips" / "clips_c.json")
    ap.add_argument("--minmax", type=Path, default=REPO / "data" / "minmax" / "summary.json")
    ap.add_argument("--template", type=Path, default=REPO / "pitch" / "app_template.html")
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "app.html")
    args = ap.parse_args(argv)
    report = Report.model_validate_json(args.report.read_text())
    site = Site.model_validate_json((SCEN / "site.json").read_text())
    curves = SensorCurves.model_validate_json((SCEN / "sensor_curve.json").read_text())
    numbers = json.loads((args.charts / "numbers.json").read_text())
    clips = json.loads(args.clips.read_text())
    fixed_name = numbers.get("before_after", {}).get("fixed") or report.baseline_config
    threats = []
    for f in sorted(args.tactics_dir.glob("top_*.json")):
        res = SearchResult.model_validate_json(f.read_text())
        t, s = res.best()
        threats.append(
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
    threats.sort(key=lambda x: -x["miss"])
    cond = report.conditions
    miss_log = (
        REPO
        / "data"
        / "clip_logs"
        / "miss"
        / f"{clips['baseline']}__{clips['tactic_id']}__{clips['seed']}.jsonl"
    )
    catch_log = (
        REPO
        / "data"
        / "clip_logs"
        / "catch"
        / f"{clips['fixed']}__{clips['tactic_id']}__{clips['seed']}.jsonl"
    )
    payload = {
        "site": site.model_dump(mode="json"),
        "curves": {
            k: {"fov_deg": c.fov_deg, "range": c.max_range_m()} for k, c in curves.curves.items()
        },
        "report": {
            "baseline": report.baseline_config,
            "fixed": fixed_name,
            "n_seeds": cond.n_seeds,
            "conditions": f"Operating point {cond.far_per_hour_operating_point:g} false alarm/h · {cond.n_seeds} seeds · {cond.sensor_calibration} · {cond.detection_model_note}",
            "configs": [
                {
                    "name": c.config_name,
                    "cost": c.cost_per_hour,
                    "pd": c.pd_at_operating_point,
                    "worst_pd": c.worst_tactic_pd,
                    "blind_pd": c.worst_tactic_pd_schedule_blind,
                    "decisions": c.human_decisions_per_hour,
                    "gap": c.coverage_gap_s_per_hour,
                }
                for c in sorted(report.configs, key=lambda c: c.cost_per_hour)
            ],
        },
        "threats": threats,
        "episodes": {
            "miss": episode(miss_log, clips["baseline"]),
            "catch": episode(catch_log, clips["fixed"]),
        },
        "images": {
            "cost": data_uri(args.charts / "cost_vs_detection.png"),
            "before_after": data_uri(args.charts / "before_after.png"),
            "map": data_uri(args.charts / "vulnerability_map.png"),
        },
        "minmax": json.loads(args.minmax.read_text()) if args.minmax.exists() else [],
    }
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = args.template.read_text().replace("__DATA__", blob)
    args.out.write_text(html)
    print(
        f"app written to {args.out} ({len(html) / 1e6:.2f} MB), minmax rows: {len(payload['minmax'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
