"""The miss and the catch: one tactic, one seed, baseline fleet versus fixed fleet, rendered to MP4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from airtight.contracts import FleetConfig, SensorCurves, Site
from airtight.redteam.search import SearchResult

REPO = Path(__file__).resolve().parents[1]
SCEN = REPO / "scenarios" / "logistics_yard"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tactics-dir", type=Path, default=REPO / "data" / "v3" / "tactics")
    ap.add_argument("--family", default="charging_window")
    ap.add_argument("--baseline", default="d2_go2_guard_sync")
    ap.add_argument("--fixed", default="d3_go2_guard_stagger")
    ap.add_argument("--site", type=Path, default=SCEN / "site.json")
    ap.add_argument("--curves", type=Path, default=SCEN / "sensor_curve.json")
    ap.add_argument("--fleets-dir", type=Path, default=SCEN / "fleets")
    ap.add_argument("--seeds", type=Path, default=REPO / "data" / "seeds.json")
    ap.add_argument(
        "--max-seeds",
        type=int,
        default=20,
        help="how many seeds to try to find a miss on the baseline that the fixed fleet catches",
    )
    ap.add_argument("--engine", default="v0")
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "clips")
    ap.add_argument("--log-dir", type=Path, default=REPO / "data" / "clip_logs")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--speed", type=float, default=3.0)
    args = ap.parse_args(argv)
    os.environ["AIRTIGHT_ENGINE"] = args.engine
    sys.path.insert(0, str(REPO / "pitch"))
    from render_replay import render

    from airtight.sim.runner import run_episode

    site = Site.model_validate_json(args.site.read_text())
    curves = SensorCurves.model_validate_json(args.curves.read_text())
    base = FleetConfig.model_validate_json((args.fleets_dir / f"{args.baseline}.json").read_text())
    fixed = FleetConfig.model_validate_json((args.fleets_dir / f"{args.fixed}.json").read_text())
    tactic = SearchResult.model_validate_json(
        (args.tactics_dir / f"top_{args.family}.json").read_text()
    ).tactics[0]
    seeds = [int(s) for s in json.loads(args.seeds.read_text())["seeds"][: args.max_seeds]]
    chosen = None
    for seed in seeds:
        miss = run_episode(site, base, tactic, curves, seed, args.log_dir / "miss")
        catch = run_episode(site, fixed, tactic, curves, seed, args.log_dir / "catch")
        if not miss.timely_detected and catch.timely_detected:
            chosen = (seed, miss, catch)
            break
    if chosen is None:
        print(
            f"no seed in the first {len(seeds)} where {args.baseline} misses and {args.fixed} catches {tactic.id}; raise --max-seeds or pick another --fixed",
            file=sys.stderr,
        )
        return 3
    seed, miss, catch = chosen
    args.out.mkdir(parents=True, exist_ok=True)
    render(miss.log_path, site, args.out / "miss.mp4", fps=args.fps, speed=args.speed)
    render(catch.log_path, site, args.out / "catch.mp4", fps=args.fps, speed=args.speed)
    (args.out / "clips.json").write_text(
        json.dumps(
            {
                "tactic_id": tactic.id,
                "family": tactic.family,
                "entry": tactic.entry_id,
                "phase": tactic.phase,
                "seed": seed,
                "baseline": args.baseline,
                "fixed": args.fixed,
                "miss_t_alarm": miss.t_alarm,
                "catch_t_alarm": catch.t_alarm,
                "t_cdp": catch.t_cdp,
            },
            indent=2,
        )
    )
    print(
        f"seed {seed}: {args.baseline} missed (alarm {miss.t_alarm}), {args.fixed} caught at {catch.t_alarm:.0f} s before the {catch.t_cdp:.0f} s deadline; clips in {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
