from __future__ import annotations

from pathlib import Path

import pytest

from airtight.sim import sanity
from airtight.sim.sanity import Cell, Job, gate, load_seeds, run_job, run_table

SEEDS_FILE = Path(__file__).parents[2] / "data" / "seeds.json"


def test_default_seeds_file_is_the_committed_one() -> None:
    assert sanity.DEFAULT_SEEDS_FILE == SEEDS_FILE
    assert len(load_seeds(SEEDS_FILE, 100)) == 100
    with pytest.raises(ValueError, match="fewer than"):
        load_seeds(SEEDS_FILE, 10_000)


def test_run_job_is_deterministic_apart_from_the_clock() -> None:
    job = Job("yard_night", "2drones", "jog", "asset", 0.0, 1000)
    a, b = run_job(job), run_job(job)
    assert (a.timely, a.alarmed) == (b.timely, b.alarmed)
    assert 0.0 < a.seconds < 1.0


def test_pool_and_serial_tables_agree() -> None:
    seeds = load_seeds(SEEDS_FILE, 3)
    serial = run_table("yard_night", seeds, workers=1, modes=("asset",))
    pooled = run_table("yard_night", seeds, workers=2, modes=("asset",))
    assert len(serial) == 3 * 3  # three tactics, three fleets, one mode
    assert [c[:6] for c in serial] == [c[:6] for c in pooled]  # everything but the timing
    assert [c.n_agents for c in serial[:3]] == [1, 2, 4]  # fleets run smallest first


def _row(
    tactic: str, mode: str, values: tuple[float, float, float], seconds: float = 0.1
) -> list[Cell]:
    fleets = (("1drone", 1), ("2drones", 2), ("4drones", 4))
    return [
        Cell(f, n, tactic, mode, v, v, seconds) for (f, n), v in zip(fleets, values, strict=True)
    ]


def test_gate_criteria() -> None:
    good = [
        *_row("walk", "asset", (0.57, 0.80, 0.97)),
        *_row("jog", "asset", (0.22, 0.45, 0.72)),
        *_row("sprint", "asset", (0.0, 0.0, 0.0)),  # no timely detection is possible: not a failure
    ]
    assert all(gate(good).values())
    falling = [*good[:3], *_row("jog", "asset", (0.30, 0.20, 0.72)), *good[6:]]
    assert [v for k, v in gate(falling).items() if k.startswith("rises")] == [False]
    weak = [*_row("walk", "asset", (0.5, 0.6, 0.84)), *good[3:]]
    assert [v for k, v in gate(weak).items() if "'walk'" in k] == [False]
    easy = [*good[:3], *_row("jog", "asset", (0.50, 0.60, 0.72)), *good[6:]]
    assert [v for k, v in gate(easy).items() if "'jog'" in k] == [False]
    slow = [*_row("walk", "asset", (0.57, 0.80, 0.97), seconds=1.5), *good[3:]]
    assert [v for k, v in gate(slow).items() if k.startswith("under")] == [False]
