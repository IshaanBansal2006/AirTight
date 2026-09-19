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


def _cmd_bakeoff(_: argparse.Namespace) -> int:
    from airtight.dimos_lane.simulator import run_h4_bakeoff

    scores = run_h4_bakeoff()
    for name, score in scores.items():
        print(
            f"{name} wins={score.wins}/3 realtime={score.near_realtime} "
            f"body={score.detector_visible_body} pose={score.pose_scriptable} "
            f"elapsed={score.elapsed_s:.2f}s"
        )
        for key, value in score.details.items():
            print(f"  {key}: {value}")
    print(f"chosen={CHOSEN_SIMULATOR}")
    print(CHOSEN_REASON)
    return 0


def _cmd_place_intruder(_: argparse.Namespace) -> int:
    from airtight.dimos_lane.person import default_intruder_xy, mujoco_start_pos, place_intruder

    xy = place_intruder()
    print(f"intruder={xy.x:.2f},{xy.y:.2f} go2_start={mujoco_start_pos()}")
    print(f"expected={default_intruder_xy().x:.2f},{default_intruder_xy().y:.2f}")
    return 0


def _cmd_yard(args: argparse.Namespace) -> int:
    from airtight.dimos_lane.person import go2_spawn_xy, mujoco_start_pos
    from airtight.dimos_lane.site_io import load_example_site
    from airtight.dimos_lane.yard import (
        occupancy_counts,
        site_to_occupancy,
        write_mujoco_occupancy_npy,
        write_occupancy_npy,
    )

    site = load_example_site()
    path = Path(args.out)
    write_occupancy_npy(site, path)
    mujoco_path = path.with_name(path.stem + "_mujoco.npy")
    write_mujoco_occupancy_npy(site, mujoco_path)
    occupied, free = occupancy_counts(site_to_occupancy(site))
    spawn = go2_spawn_xy(site)
    print(f"wrote {path} occupied={occupied} free={free} site={site.name}")
    print(f"wrote {mujoco_path} crop around {spawn.x:.1f},{spawn.y:.1f}")
    print("run with: --mujoco-room-from-occupancy " + str(mujoco_path.resolve()))
    print(f"mujoco_start_pos={mujoco_start_pos()}")
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
        example = Path(
            str(resources.files("airtight.contracts.examples").joinpath("episode.jsonl"))
        )
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

    bakeoff = sub.add_parser("bakeoff", help="run the H4 MuJoCo vs DimSim three-test bake-off")
    bakeoff.set_defaults(func=_cmd_bakeoff)

    yard = sub.add_parser("yard", help="write occupancy npy from site.json")
    yard.add_argument("--out", default="data/yard_occupancy.npy")
    yard.set_defaults(func=_cmd_yard)

    intruder = sub.add_parser("place-intruder", help="publish /person_pose in front of the Go2")
    intruder.set_defaults(func=_cmd_place_intruder)

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
