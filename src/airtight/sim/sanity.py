"""Sanity table for the v0 engine: python -m airtight.sim.sanity

For every fleet, tactic and patrol weight mode of a scenario, over the same seeds in every cell:
the fraction timely at TAU_REF, the fraction alarmed by the end of the episode, and seconds per
episode.

The part 1 gate, as accepted by the team: detection rises with fleet size in every mode, the
largest fleet on the walk is at or above 0.85, the smallest fleet on the jog (the reference
tactic) is below 0.5, and an episode takes under a second.

A second table repeats the run with the what-if EpisodeParams.task_time_s, to show the team what
a task-time field on the asset would do. It is not part of the gate.

Wall-clock time is read here to time episodes; nothing inside the simulation reads it.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from airtight.sim import scenarios
from airtight.sim.episode import EpisodeParams, simulate
from airtight.sim.geometry import WEIGHT_MODES
from airtight.sim.recorder import timely_at_ref

if TYPE_CHECKING:
    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic

DEFAULT_SEEDS_FILE = Path(__file__).resolve().parents[3] / "data" / "seeds.json"
REFERENCE_TACTIC = "jog"
EASY_TACTIC = "walk"
GATE_SMALLEST_ON_REFERENCE_BELOW = 0.5
GATE_LARGEST_ON_EASY_AT_LEAST = 0.85
GATE_SECONDS_PER_EPISODE = 1.0
WHAT_IF_TASK_TIME_S = 60.0
WHAT_IF_MODES = ("asset", "band")


class Job(NamedTuple):
    scenario: str
    fleet: str
    tactic: str
    mode: str
    task_time_s: float
    seed: int


class Outcome(NamedTuple):
    timely: bool
    alarmed: bool
    seconds: float


class Cell(NamedTuple):
    fleet: str
    n_agents: int
    tactic: str
    mode: str
    timely: float
    alarmed: float
    seconds: float


@cache
def _scenario(
    scenario: str, fleet: str, tactic: str
) -> tuple[Site, FleetConfig, Tactic, SensorCurves]:
    return (
        scenarios.load_site(scenario),
        scenarios.load_fleet(fleet, scenario),
        scenarios.load_tactic(tactic, scenario),
        scenarios.load_sensor_curves(scenario),
    )


def run_job(job: Job) -> Outcome:
    """One episode. Top level so a process pool can call it."""
    site, fleet, tactic, curves = _scenario(job.scenario, job.fleet, job.tactic)
    start = time.perf_counter()
    params = EpisodeParams(weight_mode=job.mode, task_time_s=job.task_time_s)
    scores = simulate(site, fleet, tactic, curves, job.seed, params)
    seconds = time.perf_counter() - start
    return Outcome(timely_at_ref(scores), scores.intruder_t_alarm_ref is not None, seconds)


def load_seeds(path: Path, n: int) -> list[int]:
    seeds = json.loads(path.read_text())["seeds"]
    if len(seeds) < n:
        raise ValueError(f"{path} has {len(seeds)} seeds, fewer than the {n} asked for")
    return [int(s) for s in seeds[:n]]


def run_table(
    scenario: str,
    seeds: list[int],
    workers: int | None,
    modes: tuple[str, ...] = WEIGHT_MODES,
    task_time_s: float = 0.0,
) -> list[Cell]:
    fleets = sorted(
        scenarios.names("fleet", scenario),
        key=lambda f: len(scenarios.load_fleet(f, scenario).agents),
    )
    keys = [
        (fleet, tactic, mode)
        for tactic in scenarios.names("tactic", scenario)
        for mode in modes
        for fleet in fleets
    ]
    jobs = [Job(scenario, *key, task_time_s, seed) for key in keys for seed in seeds]
    if workers == 1:
        outcomes = [run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(run_job, jobs, chunksize=len(seeds)))
    cells = []
    for i, (fleet, tactic, mode) in enumerate(keys):
        chunk = outcomes[i * len(seeds) : (i + 1) * len(seeds)]
        cells.append(
            Cell(
                fleet=fleet,
                n_agents=len(scenarios.load_fleet(fleet, scenario).agents),
                tactic=tactic,
                mode=mode,
                timely=sum(o.timely for o in chunk) / len(chunk),
                alarmed=sum(o.alarmed for o in chunk) / len(chunk),
                seconds=sum(o.seconds for o in chunk) / len(chunk),
            )
        )
    return cells


def gate(cells: list[Cell]) -> dict[str, bool]:
    """The four accepted criteria, each True or False."""
    by_row: dict[tuple[str, str], list[Cell]] = {}
    for c in cells:
        by_row.setdefault((c.tactic, c.mode), []).append(c)
    rows = {key: sorted(row, key=lambda c: c.n_agents) for key, row in by_row.items()}
    # Rising means never falling, and strictly higher at the largest fleet than the smallest.
    # A row that is 0.00 for every fleet (no timely detection is possible) has nothing to rise.
    rising = all(
        all(a.timely <= b.timely for a, b in zip(row, row[1:], strict=False))
        and (row[-1].timely > row[0].timely or row[-1].timely == 0.0)
        for row in rows.values()
    )
    return {
        "rises with fleet size in every mode": rising,
        f"largest fleet on {EASY_TACTIC!r} at or above {GATE_LARGEST_ON_EASY_AT_LEAST}": all(
            row[-1].timely >= GATE_LARGEST_ON_EASY_AT_LEAST
            for (tactic, _), row in rows.items()
            if tactic == EASY_TACTIC
        ),
        f"smallest fleet on {REFERENCE_TACTIC!r} below {GATE_SMALLEST_ON_REFERENCE_BELOW}": all(
            row[0].timely < GATE_SMALLEST_ON_REFERENCE_BELOW
            for (tactic, _), row in rows.items()
            if tactic == REFERENCE_TACTIC
        ),
        f"under {GATE_SECONDS_PER_EPISODE} s per episode": all(
            c.seconds < GATE_SECONDS_PER_EPISODE for c in cells
        ),
    }


def format_table(cells: list[Cell], n_seeds: int, title: str) -> str:
    lines = [
        title,
        f"{n_seeds} seeds per cell, the same seeds in every cell",
        f"{'tactic':8s} {'mode':8s} {'fleet':9s} {'timely':>7s} {'alarmed':>8s} {'s/episode':>10s}",
    ]
    last = None
    for c in cells:
        if last is not None and (c.tactic, c.mode) != last:
            lines.append("")
        last = (c.tactic, c.mode)
        lines.append(
            f"{c.tactic:8s} {c.mode:8s} {c.fleet:9s} {c.timely:7.2f} {c.alarmed:8.2f} {c.seconds:10.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--workers", type=int, default=None, help="1 runs without a pool")
    args = parser.parse_args(argv)

    seeds = load_seeds(args.seeds_file, args.n_seeds)
    cells = run_table(args.scenario, seeds, args.workers)
    print(format_table(cells, len(seeds), "TABLE 1: the contract as it is (no task time)"))
    verdict = gate(cells)
    print()
    for criterion, passed in verdict.items():
        print(f"gate: {'PASS' if passed else 'FAIL'}  {criterion}")
    print(f"part 1 gate: {'PASS' if all(verdict.values()) else 'FAIL'}")

    what_if = run_table(
        args.scenario, seeds, args.workers, modes=WHAT_IF_MODES, task_time_s=WHAT_IF_TASK_TIME_S
    )
    title = (
        f"TABLE 2, WHAT-IF, not part of the gate: task_time_s = {WHAT_IF_TASK_TIME_S:.0f} on the "
        "asset (a field the contract does not have)"
    )
    print()
    print(format_table(what_if, len(seeds), title))
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
