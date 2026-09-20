"""Lane A CLI: yard, calibrate, replay, sim-choice.

python -m airtight.dimos_lane yard --out data/yard_occupancy.npy
python -m airtight.dimos_lane calibrate --out data/sensor_curve.json
python -m airtight.dimos_lane replay --example --pitch pitch
python -m airtight.dimos_lane replay --example --rrd data/replay --speed 0
python -m airtight.dimos_lane replay --log path/to/episode.jsonl --live --speed 1
python -m airtight.dimos_lane replay --handoff --pitch pitch
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
    if args.live:
        from airtight.dimos_lane.calibration.capture import run_live_from_mcp

        frames = args.frames if args.frames is not None else (3 if args.live else 30)
        snapshot_dir = Path(args.snapshots) if args.snapshots else root / "calib_frames"
        n = len(run_live_from_mcp(cache, snapshot_dir, frames_per_cell=frames))
        print(f"wrote {n} live looks -> {cache}")
    elif args.synthetic or not cache.exists():
        frames = args.frames if args.frames is not None else 30
        n = len(record_synthetic_sweep(cache, seed=args.seed, frames_per_cell=frames))
        print(f"wrote {n} looks -> {cache}")
    curves = fit_cache(cache, Path(args.out))
    print(f"wrote {args.out} hash={curves.content_hash()}")
    print(curves.source)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from airtight.dimos_lane.clips import write_demo_clips
    from airtight.dimos_lane.replay import (
        dispatch_via_mcp,
        plan_replay,
        run_replay,
        write_rrd,
    )

    log: Path | None = None
    if args.handoff:
        from airtight.dimos_lane.clips import write_handoff_clips

        manifest = write_handoff_clips(Path(args.pitch), render_mp4=args.mp4)
        print(f"clips={manifest}")
        return 0
    if args.example:
        example = Path(
            str(resources.files("airtight.contracts.examples").joinpath("episode.jsonl"))
        )
        miss, catch = write_demo_clips(example, Path(args.pitch))
        print(f"miss={miss}")
        print(f"catch={catch}")
        log = example
        if args.rrd:
            dest = Path(args.rrd)
            catch_plan = plan_replay(example, dispatch=False)
            miss_log = Path(args.pitch) / "logs" / "miss.jsonl"
            if dest.suffix.lower() == ".rrd":
                write_rrd(catch_plan, dest)
                print(f"rrd={dest}")
                if miss_log.is_file():
                    miss_rrd = dest.with_name("miss.rrd")
                    write_rrd(plan_replay(miss_log, dispatch=False), miss_rrd)
                    print(f"rrd={miss_rrd}")
            else:
                dest.mkdir(parents=True, exist_ok=True)
                catch_rrd = write_rrd(catch_plan, dest / "catch.rrd")
                print(f"rrd={catch_rrd}")
                if miss_log.is_file():
                    miss_rrd = write_rrd(plan_replay(miss_log, dispatch=False), dest / "miss.rrd")
                    print(f"rrd={miss_rrd}")
        if not args.live and args.speed <= 0:
            return 0
    elif args.log:
        log = Path(args.log)
    else:
        print("replay: pass --log PATH or --example")
        return 2

    assert log is not None
    plan = plan_replay(log, dispatch=not (args.live or args.rrd or args.speed > 0))
    if args.rrd and not args.example:
        dest = Path(args.rrd)
        if dest.suffix.lower() != ".rrd":
            dest = dest / f"{plan.title}.rrd"
        write_rrd(plan, dest)
        print(f"rrd={dest}")
    if args.live or args.speed > 0:
        dispatch = dispatch_via_mcp if args.live else None
        result = run_replay(
            plan,
            live=args.live,
            realtime_scale=args.speed,
            dispatch=dispatch,
        )
        print(
            f"{result['title']} seed={plan.seed} timely={result['timely']} "
            f"person={result['person']} dispatch={result['dispatch_result']}"
        )
        return 0
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
    cal.add_argument("--live", action="store_true", help="OWLv2 + MCP snapshots + /person_pose")
    cal.add_argument("--snapshots", default=None, help="directory for live JPEG frames")
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument("--frames", type=int, default=None)
    cal.set_defaults(func=_cmd_calibrate)

    rep = sub.add_parser("replay", help="plan a replay or write pitch clips")
    rep.add_argument("--log", default=None)
    rep.add_argument("--example", action="store_true")
    rep.add_argument("--pitch", default="pitch")
    rep.add_argument(
        "--live", action="store_true", help="publish /person_pose and MCP dispatch_verify"
    )
    rep.add_argument("--rrd", default=None, help="write a Rerun .rrd (file or directory)")
    rep.add_argument("--speed", type=float, default=0.0, help="realtime scale; 1=wall clock")
    rep.add_argument(
        "--handoff",
        action="store_true",
        help="A9: same-seed miss/catch clips in pitch/clips for lane C",
    )
    rep.add_argument(
        "--mp4",
        action="store_true",
        help="with --handoff, also render miss/catch MP4s (slow; needs ffmpeg)",
    )
    rep.set_defaults(func=_cmd_replay)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
