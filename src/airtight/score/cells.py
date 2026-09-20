"""The campaign's evaluator: many configurations through one process pool, cached on disk.

A configuration here is a site variant, a fleet and the engine parameters of its policy. The
sweep caches one file per cell and starts a pool per configuration, which is right for a dozen
configurations and wrong for thousands of candidates. Here each configuration has ONE
append-only file, <cache>/<engine_tag>/<site_hash>/<fleet_hash>.jsonl. A row is one episode,
keyed by (tactic content hash, seed), or one quiet night, keyed by seed. Only the lead process
writes. A line that does not parse (the last one, after a crash) is skipped and recomputed.

Work is cut into seed chunks and submitted chunk by chunk, so when a deadline stops a run the
seeds finished for EVERY cell form a prefix, and the stage can still report on that prefix.

Seeds carry a role (seedsplit). run refuses a seed from outside the role it was given, which is
how the campaign keeps selection and reporting apart.
"""

from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from airtight.score.sweep import engine_tag
from airtight.sim.episode import EpisodeParams, simulate, simulate_quiet

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
    from airtight.score.seedsplit import SeedSplit

    Array = npt.NDArray[np.float64]
    Quiet = list[tuple[Array, float]]

CACHE_CAP_BYTES = 3 * 1024**3
CHUNK_SEEDS = 10
QUIET_KEY = "quiet"


class CacheFull(RuntimeError):  # noqa: N818  it is a stop signal, not a fault
    """The cache reached its cap. Nothing more is simulated."""


@dataclass(frozen=True)
class Config:
    name: str
    site: Site
    fleet: FleetConfig
    params: EpisodeParams

    def key(self, curves: SensorCurves) -> str:
        tag = engine_tag(self.params, curves)
        return f"{tag}/{self.site.content_hash()}/{self.fleet.content_hash()}"


@dataclass(frozen=True)
class Request:
    config: Config
    tactics: tuple[Tactic, ...]
    seeds: tuple[int, ...]
    quiet_seeds: tuple[int, ...] = ()


EpisodeTask = tuple["Site", "FleetConfig", "Tactic", "SensorCurves", tuple[int, ...], EpisodeParams]
QuietTask = tuple["Site", "FleetConfig", "SensorCurves", int, EpisodeParams]


def episode_task(task: EpisodeTask) -> list[dict[str, Any]]:
    """Some seeds of one cell. Top level so a process pool can call it."""
    site, fleet, tactic, curves, seeds, params = task
    return [dataclasses.asdict(simulate(site, fleet, tactic, curves, s, params)) for s in seeds]


def quiet_task(task: QuietTask) -> list[dict[str, Any]]:
    site, fleet, curves, seed, params = task
    return [dataclasses.asdict(simulate_quiet(site, fleet, curves, seed, params=params))]


def tactic_cost_s(site: Site, tactic: Tactic, params: EpisodeParams) -> float:
    """Simulated seconds in one episode of this tactic: what the budget is counted in."""
    pts = [site.entry(tactic.entry_id).position, *tactic.waypoints]
    xy = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    return params.warmup_s + length / tactic.speed_mps + params.task_time_s + params.tail_s


def quiet_cost_s(fleet: FleetConfig, params: EpisodeParams) -> float:
    cycles = [a.endurance_s + a.charge_time_s for a in fleet.agents if a.charge_time_s > 0]
    return params.warmup_s + (float(np.mean(cycles)) if cycles else 3600.0)


class Evaluator:
    def __init__(
        self,
        cache_dir: Path,
        curves: SensorCurves,
        seeds: SeedSplit | None,
        workers: int,
        cap_bytes: int = CACHE_CAP_BYTES,
    ) -> None:
        self.cache_dir = cache_dir
        self.curves = curves
        self.seeds = seeds
        self.workers = max(1, workers)
        self.cap_bytes = cap_bytes
        self.n_episodes = 0  # simulated by this process, not read from the cache
        self.n_quiet = 0
        self.sim_seconds = 0.0
        self.busy_s = 0.0
        self._rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
        self._pool: ProcessPoolExecutor | None = None
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._bytes = sum(p.stat().st_size for p in cache_dir.rglob("*.jsonl"))

    # ---- lifecycle -------------------------------------------------------------------------

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @property
    def cache_bytes(self) -> int:
        return self._bytes

    # ---- cache -----------------------------------------------------------------------------

    def _path(self, config: Config) -> Path:
        return self.cache_dir / f"{config.key(self.curves)}.jsonl"

    def _load(self, config: Config) -> dict[tuple[str, int], dict[str, Any]]:
        key = config.key(self.curves)
        if key not in self._rows:
            rows: dict[tuple[str, int], dict[str, Any]] = {}
            path = self._path(config)
            if path.is_file():
                for line in path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                        rows[(str(row.pop("_k")), int(row["seed"]))] = row
                    except (ValueError, KeyError, TypeError, AttributeError):
                        continue  # a torn line from a crash; the seed is simply recomputed
            self._rows[key] = rows
        return self._rows[key]

    def forget(self, config: Config) -> None:
        """Drop a configuration's rows from memory. They stay on disk."""
        self._rows.pop(config.key(self.curves), None)

    def _append(self, config: Config, kind: str, new_rows: Sequence[dict[str, Any]]) -> None:
        lines = [json.dumps({**row, "_k": kind}, sort_keys=True) for row in new_rows]
        text = "".join(line + "\n" for line in lines)
        path = self._path(config)
        if path.is_file() and path.stat().st_size:
            with path.open("rb") as handle:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    text = "\n" + text  # a crash left a torn line; never glue a row onto it
        if self._bytes + len(text) > self.cap_bytes:
            raise CacheFull(f"the cache would exceed {self.cap_bytes} bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(text)
        self._bytes += len(text)
        rows = self._load(config)
        for line in lines:
            # keep exactly what a later load would read, so a fresh run and a resumed one agree
            stored = json.loads(line)
            stored.pop("_k")
            rows[(kind, int(stored["seed"]))] = stored

    # ---- running ---------------------------------------------------------------------------

    def run(
        self, requests: Sequence[Request], role: str | None, stop_at: float | None = None
    ) -> int:
        """Simulate whatever the requests need that the cache lacks. Returns n such that the
        first n seeds of every request are complete for every cell: the longest request's
        length unless stop_at (time.time()) cut the run short, so take seeds[:n] per request.
        Quiet nights are submitted first and are never cut: a prefix of intrusion seeds with
        no operating point would be useless."""
        if self.seeds is not None and role is not None:
            for request in requests:
                self.seeds.check_role(role, request.seeds, request.quiet_seeds)
        elif self.seeds is not None:
            raise ValueError("an evaluator with a seed split needs a role for every run")
        started = time.time()
        quiet_jobs: list[tuple[Config, str, Any, float]] = []
        chunks: dict[int, list[tuple[Config, str, Any, float]]] = {}
        # (configuration, kind, seed) already claimed by an earlier request of this run, so
        # two requests that overlap never simulate a row twice
        claimed: set[tuple[str, str, int]] = set()
        for request in requests:
            config = request.config
            key = config.key(self.curves)
            rows = self._load(config)
            for seed in request.quiet_seeds:
                if (QUIET_KEY, seed) not in rows and (key, QUIET_KEY, seed) not in claimed:
                    claimed.add((key, QUIET_KEY, seed))
                    night = (config.site, config.fleet, self.curves, seed, config.params)
                    cost = quiet_cost_s(config.fleet, config.params)
                    quiet_jobs.append((config, QUIET_KEY, night, cost))
            for tactic in request.tactics:
                kind = tactic.content_hash()
                cost = tactic_cost_s(config.site, tactic, config.params)
                for start in range(0, len(request.seeds), CHUNK_SEEDS):
                    part = request.seeds[start : start + CHUNK_SEEDS]
                    missing = tuple(
                        s for s in part if (kind, s) not in rows and (key, kind, s) not in claimed
                    )
                    if missing:
                        claimed.update((key, kind, s) for s in missing)
                        task = (
                            config.site,
                            config.fleet,
                            tactic,
                            self.curves,
                            missing,
                            config.params,
                        )
                        chunks.setdefault(start // CHUNK_SEEDS, []).append(
                            (config, kind, task, cost * len(missing))
                        )
        ordered = list(quiet_jobs)
        chunk_of: list[int] = [-1] * len(ordered)
        for index in sorted(chunks):
            for job in chunks[index]:
                ordered.append(job)
                chunk_of.append(index)
        n_seeds_max = max((len(r.seeds) for r in requests), default=0)
        if not ordered:
            return n_seeds_max
        try:
            cut_chunk = self._execute(ordered, chunk_of, stop_at)
        except BaseException:
            self.close()  # a dead worker breaks the pool for good; the next run starts a new one
            raise
        self.busy_s += time.time() - started
        if cut_chunk is None:
            return n_seeds_max
        return min(n_seeds_max, cut_chunk * CHUNK_SEEDS)

    def _finish(self, job: tuple[Config, str, Any, float], new_rows: list[dict[str, Any]]) -> None:
        config, kind, _, cost = job
        self._append(config, kind, new_rows)
        self.sim_seconds += cost
        if kind == QUIET_KEY:
            self.n_quiet += len(new_rows)
        else:
            self.n_episodes += len(new_rows)

    def _execute(
        self,
        jobs: list[tuple[Config, str, Any, float]],
        chunk_of: list[int],
        stop_at: float | None,
    ) -> int | None:
        """Run the jobs in order with a bounded window of submitted work. Returns None when
        everything ran, else the index of the first seed chunk that is not complete."""

        def fn_for(kind: str) -> Any:
            return quiet_task if kind == QUIET_KEY else episode_task

        if self.workers == 1:
            for i, job in enumerate(jobs):
                if stop_at is not None and chunk_of[i] >= 0 and time.time() > stop_at:
                    return chunk_of[i]
                self._finish(job, fn_for(job[1])(job[2]))
            return None
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        pending: dict[Future[list[dict[str, Any]]], int] = {}
        next_job = 0
        stopped = False
        window = self.workers * 3
        while next_job < len(jobs) or pending:
            while not stopped and next_job < len(jobs) and len(pending) < window:
                if stop_at is not None and chunk_of[next_job] >= 0 and time.time() > stop_at:
                    stopped = True
                    break
                job = jobs[next_job]
                pending[self._pool.submit(fn_for(job[1]), job[2])] = next_job
                next_job += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                self._finish(jobs[pending.pop(future)], future.result())
        if not stopped:
            return None
        # jobs are ordered by chunk, so everything before the first unsubmitted job's chunk ran
        return chunk_of[next_job]

    # ---- reading ---------------------------------------------------------------------------

    def peaks(self, config: Config, tactics: Sequence[Tactic], seeds: Sequence[int]) -> Array:
        """intruder_peak, shape (n_seeds, n_tactics). Raises if a cell is not in the cache."""
        rows = self._load(config)
        out = np.empty((len(seeds), len(tactics)), dtype=np.float64)
        for j, tactic in enumerate(tactics):
            kind = tactic.content_hash()
            for i, seed in enumerate(seeds):
                row = rows.get((kind, seed))
                if row is None:
                    raise KeyError(f"{config.name}: tactic {tactic.id!r} seed {seed} is not cached")
                out[i, j] = row["intruder_peak"]
        return out

    def quiet(self, config: Config, quiet_seeds: Sequence[int]) -> Quiet:
        rows = self._load(config)
        out: Quiet = []
        for seed in quiet_seeds:
            row = rows.get((QUIET_KEY, seed))
            if row is None:
                raise KeyError(f"{config.name}: quiet night {seed} is not cached")
            peaks = np.array(list(row["benign_peaks"].values()), dtype=np.float64)
            out.append((peaks, float(row["sim_hours"])))
        return out

    def row(self, config: Config, kind: str, seed: int) -> dict[str, Any] | None:
        return self._load(config).get((kind, seed))

    def cached_keys(self, config: Config) -> list[tuple[str, int]]:
        return sorted(self._load(config))
