from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from airtight.contracts import FleetConfig, SensorCurves, Site
from airtight.redteam.search import load_seeds
from airtight.score.logreport.logs import EpisodeSummary
from airtight.score.logreport.sweep import (
    ConfigInputs,
    build_report,
    coverage_gap,
    inputs_without_quiet,
    load_top_tactics,
    run_config,
    run_quiet_nights,
    seed_list_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SCEN = REPO_ROOT / "scenarios" / "logistics_yard"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="airtight-sweep",
        description="Run every fleet configuration against the top tactics and write report.json.",
    )
    ap.add_argument("--site", type=Path, default=SCEN / "site.json")
    ap.add_argument("--curves", type=Path, default=SCEN / "sensor_curve.json")
    ap.add_argument("--fleets-dir", type=Path, default=SCEN / "fleets")
    ap.add_argument("--sweep", type=Path, default=SCEN / "fleets" / "sweep.json")
    ap.add_argument("--tactics-dir", type=Path, default=REPO_ROOT / "data" / "v0" / "tactics")
    ap.add_argument("--per-family", type=int, default=2, help="top tactics per family to sweep")
    ap.add_argument("--seeds", type=Path, default=REPO_ROOT / "data" / "seeds.json")
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument(
        "--quiet-seeds",
        type=int,
        default=20,
        help="quiet nights per configuration, taken from the end of the seed list; 0 falls back to benign peaks inside intrusion episodes",
    )
    ap.add_argument(
        "--far", type=float, default=1.0, help="operating point, false alarms per benign hour"
    )
    ap.add_argument("--engine", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--log-dir", type=Path, default=REPO_ROOT / "data" / "sweep_logs")
    ap.add_argument(
        "--keep-logs",
        action="store_true",
        help="keep every episode log (large); default prunes after summarising",
    )
    ap.add_argument(
        "--no-schedule-blind",
        action="store_true",
        help="skip the second pass where the same tactics get random entry phases",
    )
    ap.add_argument("--only", nargs="*", default=None, help="subset of config names")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "report.json")
    ap.add_argument(
        "--detail-dir",
        type=Path,
        default=None,
        help="per-config episode summaries, quiet stats and coverage gap; default <out>_detail/",
    )
    ap.add_argument(
        "--sensor-calibration", default="hand-written stub curve with a 360-degree drone disc"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.engine:
        os.environ["AIRTIGHT_ENGINE"] = args.engine
    engine = os.environ.get("AIRTIGHT_ENGINE", "stub")
    from airtight.sim.runner import run_episode

    site = Site.model_validate_json(args.site.read_text())
    curves = SensorCurves.model_validate_json(args.curves.read_text())
    sweep = json.loads(args.sweep.read_text())
    names = [n for n in sweep["configs"] if not args.only or n in args.only]
    if sweep["baseline"] not in names:
        names.insert(0, sweep["baseline"])
    tactics = load_top_tactics(args.tactics_dir, args.per_family)
    seeds = load_seeds(args.seeds, args.n_seeds)
    all_seeds = json.loads(args.seeds.read_text())["seeds"]
    quiet_seeds = [int(s) for s in all_seeds[-args.quiet_seeds :]] if args.quiet_seeds > 0 else []
    per_config: dict[str, ConfigInputs] = {}
    for name in names:
        fleet = FleetConfig.model_validate_json((args.fleets_dir / f"{name}.json").read_text())
        summaries = run_config(
            site,
            fleet,
            tactics,
            curves,
            seeds,
            run_episode,
            args.log_dir,
            args.workers,
            prune_logs=not args.keep_logs,
        )
        blind: list[EpisodeSummary] = []
        if not args.no_schedule_blind:
            blind = run_config(
                site,
                fleet,
                tactics,
                curves,
                seeds,
                run_episode,
                args.log_dir,
                args.workers,
                prune_logs=not args.keep_logs,
                randomize_phase=True,
            )
        if quiet_seeds and engine != "stub":
            quiet = run_quiet_nights(site, fleet, curves, quiet_seeds, args.workers)
            gap = coverage_gap(site, fleet, curves)
            per_config[name] = ConfigInputs(
                fleet=fleet,
                summaries=summaries,
                quiet=quiet,
                coverage_gap_s_per_hour=gap,
                summaries_blind=blind,
            )
        else:
            per_config[name] = inputs_without_quiet(fleet, summaries).model_copy(
                update={"summaries_blind": blind}
            )
        inp = per_config[name]
        print(
            f"{name:28s} {len(summaries):5d} episodes  timely@ref={sum(s.timely_at_ref for s in summaries) / len(summaries):.2f}"
            f"  quiet={inp.quiet.hours:.1f} h  gap={inp.coverage_gap_s_per_hour:.0f} s/h",
            file=sys.stderr,
        )
    report = build_report(
        site,
        per_config,
        sweep["baseline"],
        args.far,
        seeds,
        seed_list_hash(seeds),
        {
            "adversary_knowledge": "open-loop adversary with full knowledge of the patrol policy and charge schedule; search plus LLM proposals",
            "sensor_calibration": args.sensor_calibration,
            "detection_model_note": f"reduced-order per-look Bernoulli model, truth association, engine {engine}; false alarms from {next(iter(per_config.values())).quiet.source}",
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json(indent=2))
    detail = args.detail_dir or args.out.with_name(args.out.stem + "_detail")
    detail.mkdir(parents=True, exist_ok=True)
    for name, inp in per_config.items():
        (detail / f"{name}.json").write_text(inp.model_dump_json())
    (detail / "run.json").write_text(
        json.dumps(
            {
                "engine": engine,
                "site": str(args.site),
                "curves": str(args.curves),
                "tactics_dir": str(args.tactics_dir),
                "per_family": args.per_family,
                "n_seeds": len(seeds),
                "quiet_seeds": quiet_seeds,
                "far_target": args.far,
                "tactic_ids": [t.id for t in tactics],
            },
            indent=2,
        )
    )
    for c in report.configs:
        print(
            f"{c.config_name:28s} pd@op={c.pd_at_operating_point:.2f} [{c.pd_at_operating_point_ci[0]:.2f},{c.pd_at_operating_point_ci[1]:.2f}]  worst={c.worst_tactic_pd:.2f} blind={c.worst_tactic_pd_schedule_blind if c.worst_tactic_pd_schedule_blind is not None else float('nan'):.2f} ({c.worst_tactic_id})  cost=${c.cost_per_hour:.0f}/h  decisions/h={c.human_decisions_per_hour:.2f}"
        )
    print(
        f"engine={engine}; {len(tactics)} tactics x {len(seeds)} seeds x {len(names)} configs; report written to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
