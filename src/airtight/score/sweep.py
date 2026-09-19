"""The sweep: every configuration against every tactic over the same seeds, cached and resumable.

    python -m airtight.score.sweep --scenario-dir <dir> --seeds 50 --quiet-seeds 20

A scenario directory uses lane C's layout: site.json, sensor_curve.json, redteam_config.json,
and a fleets/ folder holding one file per configuration plus sweep.json, which names the
baseline and lists the configurations in order: {"baseline": "<name>", "configs": ["<name>", ...]}.

Tactics come from the directories given with --tactics-dir. A tactic file holds one Tactic or a
list of them. With no tactic files there is a stand-in adversary: for every entry on the site a
straight line to the asset at the scenario's speed cap, at GRID_PHASES evenly spaced phases,
plus the midpoint of every gap of every configuration in the sweep (the stretches with nobody on
duty, and the stretches with no drone on duty). A grid alone steps over gaps narrower than its
spacing. The union is taken over all configurations, so every configuration faces the same full
set, comparisons stay paired, and no configuration's own gap goes unattacked. load_inputs fails
loudly if any gap of any configuration still holds no tactic phase.

A cell is one configuration and one tactic. Its results live in
<out>/<engine_tag>/<site_hash>/<fleet_hash>/<tactic_id>.jsonl, one EpisodeScores per line, and a
configuration's quiet nights in .../<fleet_hash>/quiet/quiet.jsonl. engine_tag joins
ENGINE_VERSION with short hashes of official_params() and of the sensor curve, so results from a
different engine, parameters or curve can never be mixed in. Files are written atomically and a
rerun only computes the seeds that are missing, so an interrupted sweep resumes.

Episodes always run through simulate and simulate_quiet with official_params(), never
run_episode. Intrusion seeds are the first N of the seed list and quiet seeds the last M.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
from airtight.score.quick import DEFAULT_SEEDS_FILE, split_seeds
from airtight.sim.constants import ENGINE_VERSION
from airtight.sim.coverage import uncovered_intervals
from airtight.sim.episode import (
    EpisodeParams,
    EpisodeScores,
    QuietScores,
    official_params,
    simulate,
    simulate_quiet,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

GRID_PHASES = 16
GRID_SOURCE = "stand-in grid"
PHASE_DEDUP = 0.01  # a gap midpoint this close to a phase already in the set adds nothing
SWEEP_SPEC = "sweep.json"
REDTEAM_CONFIG = "redteam_config.json"
PROBE_EPISODES = 20


@dataclass(frozen=True)
class SweepInputs:
    site: Site
    curves: SensorCurves
    fleets: dict[str, FleetConfig]  # in the sweep spec's order
    baseline: str
    tactics: list[Tactic]
    tactic_source: str
    speed_cap_mps: float | None


@dataclass(frozen=True)
class SweepResult:
    inputs: SweepInputs
    engine_tag: str
    seeds: list[int]
    quiet_seeds: list[int]
    episodes: dict[str, dict[str, list[EpisodeScores]]]  # config -> tactic id -> seed order
    quiet: dict[str, list[QuietScores]]  # config -> quiet-seed order
    n_computed: int  # episodes and quiet nights simulated in this run; the rest came from cache


# ---- loading and validation ------------------------------------------------------------------


def _read(path: Path, problems: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        problems.append(f"{path}: file not found")
    except json.JSONDecodeError as err:
        problems.append(f"{path}: not valid JSON ({err})")
    return None


def _validate(model: type[Any], data: Any, path: Path, problems: list[str]) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as err:
        first = err.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        problems.append(
            f"{path}: not a valid {model.__name__} ({err.error_count()} errors; first: "
            f"{where}: {first['msg']})"
        )
    return None


def speed_cap(scenario_dir: Path) -> float | None:
    path = scenario_dir / REDTEAM_CONFIG
    if not path.is_file():
        return None
    value = json.loads(path.read_text()).get("speed_cap_mps")
    return None if value is None else float(value)


def grid_tactics(site: Site, speed_mps: float, n_phases: int = GRID_PHASES) -> list[Tactic]:
    """The stand-in adversary: every entry, straight to the asset, at evenly spaced phases."""
    return [
        Tactic(
            id=f"grid-{entry.id}-{k / n_phases:.4f}",
            family="charging_window",
            entry_id=entry.id,
            phase=k / n_phases,
            speed_mps=speed_mps,
            waypoints=[site.asset],
            origin="hand",
        )
        for entry in site.entry_points
        for k in range(n_phases)
    ]


def fleet_gaps(fleet: FleetConfig) -> dict[str, list[tuple[float, float]]]:
    """A configuration's gaps as phase ranges: nobody on duty, and no drone on duty."""
    drones = [a.id for a in fleet.agents if a.type == "drone"]
    return {
        "uncovered": [(g.start_phase, g.end_phase) for g in uncovered_intervals(fleet)],
        "drones_down": [
            (g.start_phase, g.end_phase) for g in uncovered_intervals(fleet, only=drones)
        ],
    }


def _in_gap(phase: float, gap: tuple[float, float]) -> bool:
    return gap[0] <= phase < gap[1]


def gap_phases(fleets: Sequence[FleetConfig], base_phases: Sequence[float]) -> list[float]:
    """Midpoints of every gap of every configuration, not within PHASE_DEDUP of a phase already
    in the set. A gap still empty after that (it can only be one narrower than 2 * PHASE_DEDUP
    whose midpoint sat next to a phase just outside it) gets its midpoint regardless."""
    gaps = sorted({gap for fleet in fleets for kind in fleet_gaps(fleet).values() for gap in kind})
    have = list(base_phases)
    added: list[float] = []
    for forced in (False, True):
        for gap in gaps:
            mid = (gap[0] + gap[1]) / 2.0
            if forced and any(_in_gap(p, gap) for p in have):
                continue
            if not forced and any(abs(mid - p) < PHASE_DEDUP for p in have):
                continue
            have.append(mid)
            added.append(mid)
    return sorted(added)


def stand_in_tactics(site: Site, fleets: Sequence[FleetConfig], speed_mps: float) -> list[Tactic]:
    """The grid, then the gap midpoints, for every entry. Ids grid-<entry>-<phase> and
    gap-<entry>-<phase>."""
    grid = grid_tactics(site, speed_mps)
    extra = gap_phases(fleets, sorted({t.phase for t in grid}))
    gaps = [
        grid[0].model_copy(
            update={"id": f"gap-{entry.id}-{phase:.4f}", "entry_id": entry.id, "phase": phase}
        )
        for entry in site.entry_points
        for phase in extra
    ]
    return [*grid, *gaps]


def unattacked_gaps(fleets: dict[str, FleetConfig], tactics: Sequence[Tactic]) -> list[str]:
    """One line per gap, of any configuration, that holds no tactic phase."""
    phases = sorted({t.phase for t in tactics})
    return [
        f"{name}: {kind} gap {gap[0]:.3f}-{gap[1]:.3f} holds no tactic phase"
        for name, fleet in fleets.items()
        for kind, gaps in fleet_gaps(fleet).items()
        for gap in gaps
        if not any(_in_gap(p, gap) for p in phases)
    ]


def load_inputs(
    scenario_dir: Path, tactic_dirs: Sequence[Path] = (), speed_cap_mps: float | None = None
) -> SweepInputs:
    """Read and validate everything. Every bad file is reported in one ValueError."""
    problems: list[str] = []
    site = _validate(
        Site, _read(scenario_dir / "site.json", problems), scenario_dir / "site.json", problems
    )
    curve_path = scenario_dir / "sensor_curve.json"
    curves = _validate(SensorCurves, _read(curve_path, problems), curve_path, problems)

    spec_path = scenario_dir / "fleets" / SWEEP_SPEC
    spec = _read(spec_path, problems)
    fleets: dict[str, FleetConfig] = {}
    baseline = ""
    if spec is not None:
        if not isinstance(spec, dict) or not {"baseline", "configs"} <= set(spec):
            problems.append(f'{spec_path}: expected {{"baseline": name, "configs": [names]}}')
        else:
            baseline = str(spec["baseline"])
            for name in spec["configs"]:
                path = scenario_dir / "fleets" / f"{name}.json"
                data = _read(path, problems)
                fleet = None if data is None else _validate(FleetConfig, data, path, problems)
                if fleet is not None:
                    fleets[str(name)] = fleet
            if baseline not in spec["configs"]:
                problems.append(f"{spec_path}: baseline {baseline!r} is not in configs")

    tactics: list[Tactic] = []
    seen: dict[str, Path] = {}
    for directory in tactic_dirs:
        if not directory.is_dir():
            problems.append(f"{directory}: tactic directory not found")
            continue
        for path in sorted(directory.glob("*.json")):
            data = _read(path, problems)
            for item in data if isinstance(data, list) else [] if data is None else [data]:
                tactic = _validate(Tactic, item, path, problems)
                if tactic is None:
                    continue
                if tactic.id in seen:
                    problems.append(
                        f"{path}: tactic id {tactic.id!r} already used in {seen[tactic.id]}"
                    )
                    continue
                if site is not None and tactic.entry_id not in {e.id for e in site.entry_points}:
                    problems.append(
                        f"{path}: tactic {tactic.id!r} uses unknown entry {tactic.entry_id!r}"
                    )
                    continue
                seen[tactic.id] = path
                tactics.append(tactic)

    cap = speed_cap_mps if speed_cap_mps is not None else speed_cap(scenario_dir)
    source = ", ".join(str(d) for d in tactic_dirs)
    if not tactics and site is not None and not problems:
        if cap is None:
            problems.append(
                f"no tactic files and no speed cap: give --speed-cap or put speed_cap_mps in "
                f"{scenario_dir / REDTEAM_CONFIG}"
            )
        else:
            tactics = stand_in_tactics(site, list(fleets.values()), cap)
            n_phases = len({t.phase for t in tactics})
            source = (
                f"{GRID_SOURCE}: {len(site.entry_points)} entries x {n_phases} phases "
                f"({GRID_PHASES} evenly spaced + {n_phases - GRID_PHASES} gap midpoints) at {cap:g} m/s"
            )
            missed = unattacked_gaps(fleets, tactics)
            if missed:  # cannot happen by construction; if it does, no number may be trusted
                raise RuntimeError(
                    "the stand-in adversary leaves gaps unattacked:\n  - " + "\n  - ".join(missed)
                )
    if problems:
        raise ValueError("sweep inputs are invalid:\n  - " + "\n  - ".join(problems))
    return SweepInputs(site, curves, fleets, baseline, tactics, source, cap)


# ---- the cache -------------------------------------------------------------------------------


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def engine_tag(params: EpisodeParams, curves: SensorCurves) -> str:
    fields = json.dumps(dataclasses.asdict(params), sort_keys=True)
    return f"v{ENGINE_VERSION}-p{_short_hash(fields)}-c{curves.content_hash()[:8]}"


def config_dir(out: Path, inputs: SweepInputs, tag: str, config: str) -> Path:
    return out / tag / inputs.site.content_hash() / inputs.fleets[config].content_hash()


def _write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(line + "\n" for line in lines))
    tmp.replace(path)


def _load_lines(path: Path, guard: str) -> dict[int, dict[str, Any]]:
    """Cached rows by seed. Rows written for a different tactic or fleet content are dropped."""
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.pop("_guard", None) == guard:
                rows[int(row["seed"])] = row
    return rows


def _dump(rows: dict[int, dict[str, Any]], order: Sequence[int], guard: str) -> list[str]:
    """Requested seeds first, in their order, then any other cached seeds, ascending."""
    wanted = [s for s in order if s in rows]
    extra = sorted(set(rows) - set(wanted))
    return [json.dumps({**rows[s], "_guard": guard}, sort_keys=True) for s in [*wanted, *extra]]


# ---- jobs ------------------------------------------------------------------------------------

CellJob = tuple[Site, FleetConfig, Tactic, SensorCurves, tuple[int, ...]]
QuietJob = tuple[Site, FleetConfig, SensorCurves, tuple[int, ...]]


def _cell_job(job: CellJob) -> list[dict[str, Any]]:
    """The missing seeds of one cell. Top level so a process pool can call it."""
    site, fleet, tactic, curves, seeds = job
    params = official_params()
    return [dataclasses.asdict(simulate(site, fleet, tactic, curves, s, params)) for s in seeds]


def _quiet_job(job: QuietJob) -> list[dict[str, Any]]:
    site, fleet, curves, seeds = job
    params = official_params()
    return [
        dataclasses.asdict(simulate_quiet(site, fleet, curves, s, params=params)) for s in seeds
    ]


def grid_hits(inputs: SweepInputs) -> dict[str, dict[str, Any]]:
    """For each configuration: its uncovered intervals, the intervals with every DRONE down,
    and how many distinct tactic phases fall inside each. The check on the grid's claim."""
    phases = sorted({t.phase for t in inputs.tactics})
    out = {}
    for name, fleet in inputs.fleets.items():
        row: dict[str, Any] = {}
        for key, gaps in fleet_gaps(fleet).items():
            row[key] = gaps
            row[f"{key}_phases_hit"] = sum(any(_in_gap(p, g) for g in gaps) for p in phases)
            row[f"{key}_gaps_missed"] = sum(not any(_in_gap(p, g) for p in phases) for g in gaps)
        out[name] = row
    return out


def run_sweep(
    inputs: SweepInputs,
    seeds: Sequence[int],
    quiet_seeds: Sequence[int],
    out: Path,
    workers: int | None = None,
    verbose: bool = False,
) -> SweepResult:
    seeds, quiet_seeds = [int(s) for s in seeds], [int(s) for s in quiet_seeds]
    params = official_params()
    tag = engine_tag(params, inputs.curves)

    def say(text: str) -> None:
        if verbose:
            print(text, flush=True)

    # what is missing
    plan: list[tuple[str, Path, str, Any, dict[int, dict[str, Any]], tuple[int, ...]]] = []
    for config, fleet in inputs.fleets.items():
        base = config_dir(out, inputs, tag, config)
        for tactic in inputs.tactics:
            guard = f"{fleet.content_hash()}:{tactic.content_hash()}"
            path = base / f"{tactic.id}.jsonl"
            rows = _load_lines(path, guard)
            missing = tuple(s for s in seeds if s not in rows)
            plan.append((config, path, guard, tactic, rows, missing))
        guard = f"{fleet.content_hash()}:quiet"
        path = base / "quiet" / "quiet.jsonl"
        rows = _load_lines(path, guard)
        missing_quiet = [s for s in quiet_seeds if s not in rows]
        # A quiet night runs a whole reference cycle, far longer than an episode, so each one is
        # its own task and the pool stays balanced. They share one rows dict and one file.
        quiet_tasks: list[tuple[int, ...]] = [(q,) for q in missing_quiet] or [()]
        for task_seeds in quiet_tasks:
            plan.append((config, path, guard, None, rows, task_seeds))

    n_episodes = sum(len(m) for _, _, _, t, _, m in plan if t is not None)
    n_quiet = sum(len(m) for _, _, _, t, _, m in plan if t is None)
    n_workers = workers or os.cpu_count() or 1
    say(f"engine tag {tag}; {len(inputs.fleets)} configurations x {len(inputs.tactics)} tactics")
    say(f"to simulate: {n_episodes} episodes and {n_quiet} quiet nights (the rest is cached)")
    if verbose and n_episodes >= PROBE_EPISODES:
        config, _, _, tactic, _, _ = next(
            p for p in plan if p[3] is not None and p[0] == inputs.baseline
        )
        start = time.perf_counter()
        _cell_job(
            (
                inputs.site,
                inputs.fleets[config],
                tactic,
                inputs.curves,
                tuple(seeds[:1] * PROBE_EPISODES),
            )
        )
        per = (time.perf_counter() - start) / PROBE_EPISODES
        per_quiet = 0.0
        if n_quiet:
            start = time.perf_counter()
            _quiet_job((inputs.site, inputs.fleets[config], inputs.curves, (quiet_seeds[0],)))
            per_quiet = time.perf_counter() - start
        total_s = (per * n_episodes + per_quiet * n_quiet) / n_workers
        say(
            f"probe on {config}: {per:.3f} s per episode, {per_quiet:.1f} s per quiet night; "
            f"about {total_s / 60:.1f} min on {n_workers} workers"
        )

    def job_for(item: Any) -> tuple[Any, Any]:
        config, _, _, tactic, _, missing = item
        fleet = inputs.fleets[config]
        if tactic is None:
            return _quiet_job, (inputs.site, fleet, inputs.curves, missing)
        return _cell_job, (inputs.site, fleet, tactic, inputs.curves, missing)

    todo = [item for item in plan if item[5]]
    done_per_config: dict[str, int] = dict.fromkeys(inputs.fleets, 0)
    todo_per_config = {c: sum(1 for i in todo if i[0] == c) for c in inputs.fleets}
    start = time.perf_counter()

    def finish(item: Any, new_rows: list[dict[str, Any]]) -> None:
        config, path, guard, tactic, rows, _ = item
        for row in new_rows:
            rows[int(row["seed"])] = row
        _write_atomic(path, _dump(rows, seeds if tactic is not None else quiet_seeds, guard))
        done_per_config[config] += 1
        if done_per_config[config] == todo_per_config[config]:
            say(f"  {config}: done ({time.perf_counter() - start:.0f} s elapsed)")

    if n_workers == 1 or len(todo) <= 1:
        for item in todo:
            fn, arg = job_for(item)
            finish(item, fn(arg))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for item in todo:
                fn, arg = job_for(item)
                futures[pool.submit(fn, arg)] = item
            for future in as_completed(futures):
                finish(futures[future], future.result())

    episodes: dict[str, dict[str, list[EpisodeScores]]] = {c: {} for c in inputs.fleets}
    quiet: dict[str, list[QuietScores]] = {}
    for config, _, _, tactic, rows, _ in plan:
        if tactic is None:
            if config not in quiet:
                quiet[config] = [QuietScores(**rows[s]) for s in quiet_seeds]
        else:
            episodes[config][tactic.id] = [EpisodeScores(**rows[s]) for s in seeds]
    return SweepResult(inputs, tag, seeds, quiet_seeds, episodes, quiet, n_episodes + n_quiet)


def _ranges(gaps: list[tuple[float, float]]) -> str:
    return ", ".join(f"{a:.3f}-{b:.3f}" for a, b in gaps) or "none"


def format_grid_hits(inputs: SweepInputs) -> str:
    lines = [
        "tactic phases inside each configuration's gaps (uncovered = nobody on duty; "
        "drones down = no drone on duty)"
    ]
    for name, row in grid_hits(inputs).items():
        missed = row["uncovered_gaps_missed"] or row["drones_down_gaps_missed"]
        warn = "   WARNING: a gap is hit by no tactic phase" if missed else ""
        lines.append(
            f"  {name:26s} uncovered [{_ranges(row['uncovered'])}] "
            f"hit by {row['uncovered_phases_hit']:2d}   "
            f"drones down [{_ranges(row['drones_down'])}] "
            f"hit by {row['drones_down_phases_hit']:2d}{warn}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--tactics-dir", type=Path, action="append", default=[])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--quiet-seeds", type=int, default=20)
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--out", type=Path, default=Path("data/sweep"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--speed-cap", type=float, default=None, help="for the stand-in adversary")
    args = parser.parse_args(argv)

    inputs = load_inputs(args.scenario_dir, args.tactics_dir, args.speed_cap)
    seeds, quiet_seeds = split_seeds(args.seeds_file, args.seeds, args.quiet_seeds)
    print(f"site {inputs.site.name}; baseline {inputs.baseline}; tactics: {inputs.tactic_source}")
    print(format_grid_hits(inputs))
    start = time.perf_counter()
    result = run_sweep(inputs, seeds, quiet_seeds, args.out, args.workers, verbose=True)
    total = len(inputs.fleets) * len(inputs.tactics) * len(seeds)
    print(
        f"sweep complete: {total} episodes in the result, {result.n_computed} simulated now, "
        f"wall time {time.perf_counter() - start:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
