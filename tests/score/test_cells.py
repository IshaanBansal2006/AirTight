from __future__ import annotations

import dataclasses
import itertools
import time
import types
from typing import TYPE_CHECKING

import numpy as np
import pytest

from airtight.score import cells, seedsplit
from airtight.score.cells import (
    CHUNK_SEEDS,
    QUIET_KEY,
    CacheFull,
    Config,
    Evaluator,
    Request,
    quiet_cost_s,
    tactic_cost_s,
)
from airtight.sim import scenarios
from airtight.sim.episode import QuietScores, official_params, simulate, simulate_quiet

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.contracts import Tactic

SPLIT = seedsplit.split(seedsplit.generate_seeds())
SEEDS = SPLIT.intrusion["search"][:12]  # two seed chunks: ten and two
QUIET_SEEDS = SPLIT.quiet["search"][:2]
SITE = scenarios.load_site()
CURVES = scenarios.load_sensor_curves()
PARAMS = official_params()


def _tactics() -> tuple[Tactic, ...]:
    jog = scenarios.load_tactic("jog")
    return (
        jog.model_copy(update={"id": "jog-up", "phase": 0.2}),
        scenarios.load_tactic("walk").model_copy(update={"id": "walk-up", "phase": 0.1}),
    )


TACTICS = _tactics()


def _config(fleet: str = "1drone", **changes: object) -> Config:
    params = dataclasses.replace(PARAMS, **changes)  # type: ignore[arg-type]
    return Config(f"{fleet}-toy", SITE, scenarios.load_fleet(fleet), params)


def _request(config: Config, n: int = len(SEEDS), quiet: bool = True) -> Request:
    return Request(config, TACTICS, SEEDS[:n], QUIET_SEEDS if quiet else ())


def _lines(root: Path) -> dict[str, list[str]]:
    return {
        str(p.relative_to(root)): p.read_text().splitlines() for p in sorted(root.rglob("*.jsonl"))
    }


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """cells sees a clock that reads 1, 2, 3, ... so a deadline falls on a known job."""
    ticks = itertools.count(1)
    monkeypatch.setattr(cells, "time", types.SimpleNamespace(time=lambda: float(next(ticks))))


def test_results_equal_direct_simulate_calls_and_the_counters_add_up(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        assert ev.run([_request(config)], "search") == len(SEEDS)
        peaks = ev.peaks(config, TACTICS, SEEDS)
        quiet = ev.quiet(config, QUIET_SEEDS)
        assert ev.n_episodes == len(SEEDS) * len(TACTICS)
        assert ev.n_quiet == len(QUIET_SEEDS)
        expected_s = len(SEEDS) * sum(tactic_cost_s(SITE, t, PARAMS) for t in TACTICS)
        expected_s += len(QUIET_SEEDS) * quiet_cost_s(config.fleet, PARAMS)
        assert ev.sim_seconds == pytest.approx(expected_s)
        assert ev.cache_bytes == sum(p.stat().st_size for p in tmp_path.rglob("*.jsonl")) > 0
    direct = [[simulate(SITE, config.fleet, t, CURVES, s, PARAMS) for t in TACTICS] for s in SEEDS]
    assert peaks.shape == (len(SEEDS), len(TACTICS))
    assert peaks.tolist() == [[e.intruder_peak for e in row] for row in direct]
    assert len(set(peaks.ravel().tolist())) > 2  # the toy cells are not all "never seen"
    for (got_peaks, got_hours), seed in zip(quiet, QUIET_SEEDS, strict=True):
        night = simulate_quiet(SITE, config.fleet, CURVES, seed, params=PARAMS)
        assert sorted(got_peaks.tolist()) == sorted(night.benign_peaks.values())
        assert len(got_peaks) > 0
        assert got_hours == night.sim_hours
    # a whole cached row is the EpisodeScores, field for field
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        row = ev.row(config, TACTICS[1].content_hash(), SEEDS[3])
        assert row == dataclasses.asdict(direct[3][1])
        assert ev.row(config, TACTICS[1].content_hash(), SPLIT.intrusion["final"][0]) is None


def test_a_fresh_evaluator_over_the_same_cache_simulates_nothing(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config)], "search")
        peaks = ev.peaks(config, TACTICS, SEEDS)
        quiet = ev.quiet(config, QUIET_SEEDS)
        keys = ev.cached_keys(config)
        assert ev.run([_request(config)], "search") == len(SEEDS)  # and nothing on a re-run
        assert ev.n_episodes == len(SEEDS) * len(TACTICS)
    before = _lines(tmp_path)
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as again:
        assert again.cache_bytes == ev.cache_bytes
        assert again.run([_request(config)], "search") == len(SEEDS)
        assert (again.n_episodes, again.n_quiet, again.sim_seconds) == (0, 0, 0.0)
        assert again.cached_keys(config) == keys
        assert len(keys) == len(SEEDS) * len(TACTICS) + len(QUIET_SEEDS)
        np.testing.assert_array_equal(again.peaks(config, TACTICS, SEEDS), peaks)
        for (a, hours_a), (b, hours_b) in zip(again.quiet(config, QUIET_SEEDS), quiet, strict=True):
            assert sorted(a.tolist()) == sorted(b.tolist())
            assert hours_a == hours_b
    assert _lines(tmp_path) == before


def test_quiet_peaks_come_back_in_the_same_order_from_memory_and_from_disk(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config)], "search")
        fresh = ev.quiet(config, QUIET_SEEDS)
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as again:
        cached = again.quiet(config, QUIET_SEEDS)
    for (a, _), (b, _) in zip(fresh, cached, strict=True):
        np.testing.assert_array_equal(a, b)


def test_quiet_peak_order_survives_the_disk_when_ids_do_not_sort_by_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def busy_night(*args: object, **kwargs: object) -> QuietScores:
        return QuietScores(QUIET_SEEDS[0], {"fox-2": 1.0, "fox-10": 2.0}, 1.0, 3600.0)

    monkeypatch.setattr(cells, "simulate_quiet", busy_night)
    config = _config()
    request = Request(config, (), (), QUIET_SEEDS[:1])
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([request], "search")
        fresh = ev.quiet(config, QUIET_SEEDS[:1])[0][0]
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as again:
        cached = again.quiet(config, QUIET_SEEDS[:1])[0][0]
    assert sorted(fresh.tolist()) == sorted(cached.tolist()) == [1.0, 2.0]
    np.testing.assert_array_equal(fresh, cached)


def test_one_and_two_workers_write_the_same_rows(tmp_path: Path) -> None:
    configs = [_config("1drone"), _config("2drones")]
    out = {}
    for workers in (1, 2):
        root = tmp_path / f"w{workers}"
        with Evaluator(root, CURVES, SPLIT, workers=workers) as ev:
            assert ev.run([_request(c) for c in configs], "search") == len(SEEDS)
            assert ev.n_episodes == 2 * len(SEEDS) * len(TACTICS)
            out[workers] = (
                {name: sorted(rows) for name, rows in _lines(root).items()},
                [ev.peaks(c, TACTICS, SEEDS).tolist() for c in configs],
            )
    assert out[1] == out[2]
    assert len(out[1][0]) == 2  # one file per configuration
    assert all(len(rows) == len(set(rows)) for rows in out[1][0].values())


def test_a_torn_last_line_is_skipped_and_recomputed(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config, quiet=False)], "search")
        peaks = ev.peaks(config, TACTICS, SEEDS)
    (path,) = tmp_path.rglob("*.jsonl")
    whole = path.read_text().splitlines()
    path.write_text("".join(line + "\n" for line in whole[:-1]) + whole[-1][: len(whole[-1]) // 2])
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        assert len(ev.cached_keys(config)) == len(whole) - 1
        assert ev.run([_request(config, quiet=False)], "search") == len(SEEDS)
        assert ev.n_episodes == 1
        np.testing.assert_array_equal(ev.peaks(config, TACTICS, SEEDS), peaks)


def test_the_cache_is_whole_again_after_a_torn_line_was_recomputed(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config, quiet=False)], "search")
    (path,) = tmp_path.rglob("*.jsonl")
    path.write_text(path.read_text()[:-40])  # the crash left no newline at the end
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config, quiet=False)], "search")
        assert ev.n_episodes == 1
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config, quiet=False)], "search")
        assert ev.n_episodes == 0  # the recomputed row must have reached the disk intact


def _with_junk(tmp_path: Path, junk: str) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config, n=2, quiet=False)], "search")
        peaks = ev.peaks(config, TACTICS, SEEDS[:2])
    (path,) = tmp_path.rglob("*.jsonl")
    with path.open("a") as handle:
        handle.write(junk)
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        assert len(ev.cached_keys(config)) == 2 * len(TACTICS)
        np.testing.assert_array_equal(ev.peaks(config, TACTICS, SEEDS[:2]), peaks)


def test_objects_that_are_not_rows_are_skipped(tmp_path: Path) -> None:
    _with_junk(tmp_path, '\n{"seed": 5}\n{"_k": "x", "seed": "many"}\n{"_k": "x"}\n{}\n')


def test_json_that_is_not_an_object_is_skipped(tmp_path: Path) -> None:
    _with_junk(tmp_path, "null\n[1, 2]\n7\n")


def test_configurations_that_differ_only_in_params_get_different_files(tmp_path: Path) -> None:
    base, other = _config(), _config(d0_m=90.0)
    assert base.key(CURVES) != other.key(CURVES)
    assert base.key(CURVES).split("/")[1:] == other.key(CURVES).split("/")[1:]
    assert _config().key(CURVES) == base.key(CURVES)  # the name plays no part
    assert dataclasses.replace(base, name="renamed").key(CURVES) == base.key(CURVES)
    assert _config("2drones").key(CURVES).split("/")[:2] == base.key(CURVES).split("/")[:2]
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(base, n=3, quiet=False), _request(other, n=3, quiet=False)], "search")
        assert ev.n_episodes == 2 * 3 * len(TACTICS)
        files = _lines(tmp_path)
        assert sorted(files) == sorted(f"{c.key(CURVES)}.jsonl" for c in (base, other))
        assert all(len(rows) == 3 * len(TACTICS) for rows in files.values())
        for config in (base, other):
            direct = [
                [simulate(SITE, config.fleet, t, CURVES, s, config.params) for t in TACTICS]
                for s in SEEDS[:3]
            ]
            got = ev.peaks(config, TACTICS, SEEDS[:3]).tolist()
            assert got == [[e.intruder_peak for e in row] for row in direct]


def test_a_deadline_already_past_still_runs_the_quiet_nights(tmp_path: Path) -> None:
    config = _config()
    for workers in (1, 2):
        with Evaluator(tmp_path / str(workers), CURVES, SPLIT, workers=workers) as ev:
            assert ev.run([_request(config)], "search", stop_at=time.time() - 60.0) == 0
            assert (ev.n_episodes, ev.n_quiet) == (0, len(QUIET_SEEDS))
            assert len(ev.quiet(config, QUIET_SEEDS)) == len(QUIET_SEEDS)
            with pytest.raises(KeyError, match="not cached"):
                ev.peaks(config, TACTICS, SEEDS[:1])
            with pytest.raises(KeyError, match="not cached"):
                ev.quiet(config, SPLIT.quiet["search"][5:6])
            assert ev.peaks(config, TACTICS, ()).shape == (0, len(TACTICS))
            # with everything cached the deadline has nothing to cut
            assert ev.run([_request(config)], "search") == len(SEEDS)
            assert ev.run([_request(config)], "search", stop_at=time.time() - 60.0) == len(SEEDS)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_deadline_cut_leaves_a_usable_seed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    config = _config()
    seeds = SPLIT.intrusion["search"][: 2 * CHUNK_SEEDS + 5]
    request = Request(config, TACTICS, seeds, QUIET_SEEDS)
    _fake_clock(monkeypatch)
    # the clock reads 1 at the start, then 2, 3, ... before each of the six chunk jobs: both
    # tactics of chunk 0 and the first of chunk 1 go ahead, the second of chunk 1 is refused
    with Evaluator(tmp_path, CURVES, SPLIT, workers=workers) as ev:
        usable = ev.run([request], "search", stop_at=4.5)
        assert usable == CHUNK_SEEDS
        assert ev.n_quiet == len(QUIET_SEEDS)
        assert ev.n_episodes == 3 * CHUNK_SEEDS
        prefix = ev.peaks(config, TACTICS, seeds[:usable])
        with pytest.raises(KeyError):
            ev.peaks(config, TACTICS, seeds[: usable + 1])
        assert ev.run([request], "search") == len(seeds)
        assert ev.n_episodes == len(seeds) * len(TACTICS)  # nothing was simulated twice
        np.testing.assert_array_equal(ev.peaks(config, TACTICS, seeds)[:usable], prefix)
        # a later cut counts the cached chunks: only the new seeds of chunk 2 are at stake
        longer = Request(config, TACTICS, SPLIT.intrusion["search"][: 3 * CHUNK_SEEDS], ())
        assert ev.run([longer], "search", stop_at=0.0) == 2 * CHUNK_SEEDS


def test_a_tiny_cap_raises_cache_full_and_writes_nothing(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1, cap_bytes=50) as ev:
        with pytest.raises(CacheFull):
            ev.run([_request(config, n=2, quiet=False)], "search")
        assert ev.cache_bytes == 0
        assert ev.n_episodes == 0
        assert ev.cached_keys(config) == []
    assert _lines(tmp_path) == {} or all(rows == [] for rows in _lines(tmp_path).values())


def test_the_cap_counts_what_is_already_on_disk_but_reading_still_works(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config, n=2, quiet=False)], "search")
        peaks = ev.peaks(config, TACTICS, SEEDS[:2])
        used = ev.cache_bytes
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1, cap_bytes=used) as ev:
        assert ev.cache_bytes == used
        assert ev.run([_request(config, n=2, quiet=False)], "search") == 2
        np.testing.assert_array_equal(ev.peaks(config, TACTICS, SEEDS[:2]), peaks)
        with pytest.raises(CacheFull):
            ev.run([_request(config, n=3, quiet=False)], "search")
        assert ev.cache_bytes == used


def test_a_seed_from_the_wrong_role_raises_before_anything_runs(tmp_path: Path) -> None:
    config = _config()
    stray = Request(config, TACTICS, (*SEEDS[:2], SPLIT.intrusion["final"][0]), ())
    stray_quiet = Request(config, TACTICS, SEEDS[:2], SPLIT.quiet["validation"][:1])
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        for role, requests in [
            ("search", [_request(config, n=2), stray]),
            ("search", [stray_quiet]),
            ("validation", [_request(config, n=2)]),
            ("final", [_request(config, n=2, quiet=False)]),
        ]:
            with pytest.raises(ValueError, match="outside its split"):
                ev.run(requests, role)
        with pytest.raises(ValueError, match="unknown seed role"):
            ev.run([_request(config, n=2)], "holdout")
        with pytest.raises(ValueError, match="needs a role"):
            ev.run([_request(config, n=2)], None)
        with pytest.raises(ValueError, match="needs a role"):
            ev.run([], None)
        assert (ev.n_episodes, ev.n_quiet, ev.cache_bytes) == (0, 0, 0)
    assert _lines(tmp_path) == {}


def test_without_a_seed_split_any_seed_runs_and_no_requests_is_no_work(tmp_path: Path) -> None:
    config = _config()
    request = Request(config, TACTICS[:1], (1000, 1001), ())
    with Evaluator(tmp_path, CURVES, None, workers=0) as ev:  # workers is floored at one
        assert ev.workers == 1
        assert ev.run([], None) == 0
        assert ev.run([request], None) == 2
        assert ev.run([request], "search") == 2
        assert ev.n_episodes == 2
        direct = [simulate(SITE, config.fleet, TACTICS[0], CURVES, s, PARAMS) for s in (1000, 1001)]
        assert ev.peaks(config, TACTICS[:1], (1000, 1001))[:, 0].tolist() == [
            e.intruder_peak for e in direct
        ]


def test_the_same_cell_asked_twice_is_simulated_once(tmp_path: Path) -> None:
    config = _config()
    twin = dataclasses.replace(config, name="the same configuration under another name")
    requests = [_request(config, n=3, quiet=True), _request(twin, n=3, quiet=True)]
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        assert ev.run(requests, "search") == 3
        assert (ev.n_episodes, ev.n_quiet) == (3 * len(TACTICS), len(QUIET_SEEDS))
    (rows,) = _lines(tmp_path).values()
    assert len(rows) == len(set(rows)) == 3 * len(TACTICS) + len(QUIET_SEEDS)


def test_overlapping_seed_lists_in_one_run_are_simulated_once(tmp_path: Path) -> None:
    config = _config()
    requests = [_request(config, n=5, quiet=False), _request(config, n=3, quiet=False)]
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        assert ev.run(requests, "search") == 5
        assert ev.n_episodes == 5 * len(TACTICS)
    (rows,) = _lines(tmp_path).values()
    assert len(rows) == len(set(rows)) == 5 * len(TACTICS)


def test_forget_drops_rows_from_memory_only(tmp_path: Path) -> None:
    config = _config()
    with Evaluator(tmp_path, CURVES, SPLIT, workers=1) as ev:
        ev.run([_request(config, n=2)], "search")
        keys = ev.cached_keys(config)
        assert {k for k, _ in keys} == {QUIET_KEY, *(t.content_hash() for t in TACTICS)}
        ev.forget(config)
        ev.forget(_config("2drones"))  # never loaded: nothing to do
        assert ev.cached_keys(config) == keys
        assert ev.run([_request(config, n=2)], "search") == 2
        assert ev.n_episodes == 2 * len(TACTICS)


def test_costs_are_the_simulated_seconds() -> None:
    fleet = scenarios.load_fleet("2drones")
    for tactic in (*TACTICS, scenarios.load_tactic("sprint")):
        scores = simulate(SITE, fleet, tactic, CURVES, SEEDS[0], PARAMS)
        assert tactic_cost_s(SITE, tactic, PARAMS) == pytest.approx(PARAMS.warmup_s + scores.t_end)
    with_task = dataclasses.replace(PARAMS, task_time_s=20.0, warmup_s=30.0)
    assert tactic_cost_s(SITE, TACTICS[0], with_task) == pytest.approx(
        tactic_cost_s(SITE, TACTICS[0], PARAMS) + 20.0 - (PARAMS.warmup_s - 30.0)
    )
    dogleg = TACTICS[0].model_copy(
        update={"waypoints": [SITE.entry(TACTICS[0].entry_id).position, *TACTICS[0].waypoints]}
    )
    assert tactic_cost_s(SITE, dogleg, PARAMS) == pytest.approx(
        tactic_cost_s(SITE, TACTICS[0], PARAMS)
    )
    night = simulate_quiet(SITE, fleet, CURVES, QUIET_SEEDS[0], params=PARAMS)
    assert quiet_cost_s(fleet, PARAMS) == pytest.approx(PARAMS.warmup_s + night.duration_s)
    no_charge = fleet.model_copy(
        update={"agents": [a.model_copy(update={"charge_time_s": 0.0}) for a in fleet.agents]}
    )
    assert quiet_cost_s(no_charge, PARAMS) == PARAMS.warmup_s + 3600.0
