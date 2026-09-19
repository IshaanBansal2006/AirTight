"""Lane A CLI: yard, calibrate, replay, sim-choice.

  python -m airtight.dimos_lane yard --out data/yard_occupancy.npy
  python -m airtight.dimos_lane calibrate --out data/sensor_curve.json
  python -m airtight.dimos_lane replay --example --pitch pitch
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from airtight.dimos_lane.simulator import CHOSEN_REASON, CHOSEN_SIMULATOR, DIMOS_TRANSPORT


def _cmd_sim(_: argparse.Namespace) -> int:
    print(f"chosen={CHOSEN_SIMULATOR}")
    print(f"transport={DIMOS_TRANSPORT}")
    print(CHOSEN_REASON)
    return 0


def _cmd_yard(args: argparse.Namespace) -> int:
    from airtight.dimos_lane.site_io import load_example_site
    from airtight.dimos_lane.yard import occupancy_counts, site_to_occupancy, write_occupancy_npy

    site = load_example_site()
    path = Path(args.out)
    write_occupancy_npy(site, path)
    occupied, free = occupancy_counts(site_to_occupancy(site))
    print(f"wrote {path} occupied={occupied} free={free} site={site.name}")
    print("run with: DIMOS_TRANSPORT=lcm DIMOS_MUJOCO_ROOM_FROM_OCCUPANCY=" + str(path.resolve()))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from airtight.dimos_lane.calibration import looks_path, record_synthetic_sweep
    from airtight.dimos_lane.calibration.fit import fit_cache

    root = Path(args.data)
    cache = Path(args.cache) if args.cache else looks_path(root)
    if args.synthetic or not cache.exists():
        n = len(record_synthetic_sweep(cache, seed=args.seed, frames_per_cell=args.frames))
        print(f"wrote {n} looks -> {cache}")
    curves = fit_cache(cache, Path(args.out))
    print(f"wrote {args.out} hash={curves.content_hash()}")
    print(curves.source)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from airtight.dimos_lane.clips import write_demo_clips
    from airtight.dimos_lane.replay import plan_replay

    if args.example:
        example = Path(str(resources.files("airtight.contracts.examples").joinpath("episode.jsonl")))
        miss, catch = write_demo_clips(example, Path(args.pitch))
        print(f"miss={miss}")
        print(f"catch={catch}")
        return 0
    plan = plan_replay(Path(args.log))
    print(
        f"{plan.title} seed={plan.seed} timely={plan.timely_detected} "
        f"dispatch={plan.dispatch_result}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airtight.dimos_lane")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("sim", help="print the H4 simulator choice")
    sim.set_defaults(func=_cmd_sim)

    yard = sub.add_parser("yard", help="write occupancy npy from site.json")
    yard.add_argument("--out", default="data/yard_occupancy.npy")
    yard.set_defaults(func=_cmd_yard)

    cal = sub.add_parser("calibrate", help="fit data/sensor_curve.json from cached looks")
    cal.add_argument("--data", default="data")
    cal.add_argument("--cache", default=None)
    cal.add_argument("--out", default="data/sensor_curve.json")
    cal.add_argument("--synthetic", action="store_true")
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument("--frames", type=int, default=30)
    cal.set_defaults(func=_cmd_calibrate)

    rep = sub.add_parser("replay", help="plan a replay or write pitch clips")
    rep.add_argument("--log", default=None)
    rep.add_argument("--example", action="store_true")
    rep.add_argument("--pitch", default="pitch")
    rep.set_defaults(func=_cmd_replay)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
