"""One command from inputs to the morning files: campaign, then report, then handoff.

    python -m airtight.score.pipeline --out /abs/path/data/campaign --deadline-hours 6

A new sensor curve or lane C's real tactics are rerun by pointing --scenario-dir and
--tactics-dir at them. The campaign's cache is keyed by the curve, the site, the fleet and the
parameters, so nothing stale can be reused, and whatever did not change is not simulated again.
A checkpoint written under another commit stops the campaign unless --allow-new-code is given;
a changed scenario wants a fresh --out instead, because the checkpoint holds the old plans.

Each step is the module's own main, so every step can also be run alone:
airtight.score.campaign, airtight.score.campaign_report, airtight.score.campaign_handoff.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

STEPS = ("campaign", "report", "handoff")
MODULES = {
    "campaign": "airtight.score.campaign",
    "report": "airtight.score.campaign_report",
    "handoff": "airtight.score.campaign_handoff",
}


def step_args(step: str, args: argparse.Namespace) -> list[str]:
    """The argument list each step's main receives."""
    out = args.out / "smoke" if args.smoke and step != "campaign" else args.out
    argv = ["--out", str(out)]
    if step != "campaign":
        return argv
    argv += ["--deadline-hours", str(args.deadline_hours)]
    if args.smoke:
        argv.append("--smoke")
    if args.scenario_dir is not None:
        argv += ["--scenario-dir", str(args.scenario_dir)]
    for path in args.tactics_dir or []:
        argv += ["--tactics-dir", str(path)]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.frozen_root is not None:
        argv += ["--frozen-root", str(args.frozen_root)]
    if args.allow_new_code:
        argv.append("--allow-new-code")
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="absolute path to data/campaign")
    parser.add_argument("--deadline-hours", type=float, default=6.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument("--tactics-dir", type=Path, action="append", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--frozen-root", type=Path, default=None)
    parser.add_argument("--allow-new-code", action="store_true")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma separated, of {STEPS}")
    args = parser.parse_args(argv)
    if not args.out.is_absolute():
        parser.error("--out must be an absolute path")
    wanted = [s for s in args.steps.split(",") if s]
    unknown = [s for s in wanted if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    for step in STEPS:
        if step not in wanted:
            continue
        print(f"pipeline: {step}", flush=True)
        code = int(importlib.import_module(MODULES[step]).main(step_args(step, args)))
        if code != 0:
            print(f"pipeline: {step} returned {code}; stopping", flush=True)
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
