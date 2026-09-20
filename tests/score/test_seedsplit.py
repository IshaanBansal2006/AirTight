from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from airtight.score import seedsplit
from airtight.score.seedsplit import (
    GENERATOR_KEY,
    INTRUSION_RANGES,
    N_SEEDS,
    QUIET_RANGES,
    ROLES,
    SeedSplit,
    assert_no_overlap,
    generate_seeds,
    load,
    split,
    write_seeds,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_generated_seeds_are_deterministic_distinct_and_in_range() -> None:
    first, second = generate_seeds(), generate_seeds()
    assert first == second
    assert len(first) == len(set(first)) == N_SEEDS == 2000
    assert all(isinstance(s, int) and 0 <= s < 2**31 - 1 for s in first)
    # the documented generator, so the list is the same on any machine
    rng = np.random.default_rng(list(GENERATOR_KEY))
    assert first[:5] == [int(rng.integers(0, 2**31 - 1)) for _ in range(5)]


def test_the_ranges_tile_the_list_without_a_gap_or_an_overlap() -> None:
    spans = sorted([*INTRUSION_RANGES.values(), *QUIET_RANGES.values()])
    assert spans[0][0] == 0 and spans[-1][1] == N_SEEDS
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert set(INTRUSION_RANGES) == set(QUIET_RANGES) == set(ROLES)


def test_split_gives_every_role_its_documented_slice() -> None:
    seeds = generate_seeds()
    parts = split(seeds)
    assert {r: len(s) for r, s in parts.intrusion.items()} == {
        "search": 200,
        "validation": 400,
        "final": 1200,
    }
    assert {r: len(s) for r, s in parts.quiet.items()} == {
        "search": 20,
        "validation": 40,
        "final": 140,
    }
    assert parts.intrusion["search"] == tuple(seeds[:200])
    assert parts.intrusion["final"] == tuple(seeds[600:1800])
    assert parts.quiet["search"] == tuple(seeds[1800:1820])
    assert parts.quiet["final"] == tuple(seeds[1860:])
    everything = [s for group in (parts.intrusion, parts.quiet) for r in ROLES for s in group[r]]
    assert sorted(everything) == sorted(seeds)
    assert_no_overlap(parts)
    assert len(parts.describe()) == 6


@pytest.mark.parametrize("n", [0, N_SEEDS - 1, N_SEEDS + 1])
def test_split_refuses_a_wrong_length(n: int) -> None:
    with pytest.raises(ValueError, match="must hold 2000"):
        split(list(range(n)))


def test_split_refuses_duplicates() -> None:
    seeds = generate_seeds()
    seeds[1999] = seeds[0]  # a final quiet seed that is also a search intrusion seed
    with pytest.raises(ValueError, match="duplicates"):
        split(seeds)


def test_any_overlap_between_two_groups_raises() -> None:
    parts = split(generate_seeds())
    shared = parts.intrusion["search"][0]
    for group, role in [("intrusion", "validation"), ("intrusion", "final"), ("quiet", "search")]:
        moved = {**getattr(parts, group), role: (*getattr(parts, group)[role], shared)}
        with pytest.raises(ValueError, match="overlap"):
            assert_no_overlap(dataclasses.replace(parts, **{group: moved}))
    same_role = SeedSplit(
        intrusion=parts.intrusion, quiet={**parts.quiet, "search": (shared, *parts.quiet["search"])}
    )
    with pytest.raises(ValueError, match="overlap"):  # intrusion and quiet of ONE role too
        assert_no_overlap(same_role)


def test_check_role_accepts_its_own_seeds_and_refuses_strays_and_unknown_roles() -> None:
    parts = split(generate_seeds())
    for role in ROLES:
        parts.check_role(role, parts.intrusion[role], parts.quiet[role])
        parts.check_role(role, parts.intrusion[role][:3], ())
        parts.check_role(role, (), ())
    with pytest.raises(ValueError, match="outside its split"):
        parts.check_role("search", [parts.intrusion["validation"][0]], [])
    with pytest.raises(ValueError, match="outside its split"):
        parts.check_role("search", [*parts.intrusion["search"], parts.intrusion["final"][7]], [])
    with pytest.raises(ValueError, match="outside its split"):
        parts.check_role("final", [], [parts.quiet["search"][0]])
    with pytest.raises(ValueError, match="outside its split"):  # a quiet seed is not an intrusion
        parts.check_role("search", [parts.quiet["search"][0]], [])
    with pytest.raises(ValueError, match="outside its split"):
        parts.check_role("search", [], [parts.intrusion["search"][0]])
    with pytest.raises(ValueError, match="outside its split"):
        parts.check_role("search", [12345678901], [])  # not a campaign seed at all
    for role in ("Search", "test", ""):
        with pytest.raises(ValueError, match="unknown seed role"):
            parts.check_role(role, [], [])


def test_load_creates_the_list_once_and_reads_it_back(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "campaign_seeds.json"
    with pytest.raises(FileNotFoundError):
        load(path)
    parts = load(path, create=True)
    assert parts == split(generate_seeds())
    body = json.loads(path.read_text())
    assert body["seeds"] == generate_seeds()
    assert str(list(GENERATOR_KEY)) in body["generator"]
    before = path.read_bytes()
    assert load(path, create=True) == parts == load(path)
    assert path.read_bytes() == before  # an existing list is never rewritten


def test_load_refuses_a_list_that_is_not_the_generators_output(tmp_path: Path) -> None:
    good = generate_seeds()
    swapped = [good[1], good[0], *good[2:]]  # same seeds, two of them in each other's place
    edited = [*good[:-1], good[-1] + 1]
    for i, seeds in enumerate([swapped, edited, list(range(N_SEEDS)), good[:-1], [*good, 5]]):
        path = tmp_path / f"seeds_{i}.json"
        path.write_text(json.dumps({"generator": "by hand", "seeds": seeds}))
        with pytest.raises(ValueError, match="is not the list"):
            load(path)
        with pytest.raises(ValueError, match="is not the list"):
            load(path, create=True)  # create never overwrites a wrong list
        assert json.loads(path.read_text())["seeds"] == seeds


def test_write_seeds_is_byte_identical_every_time(tmp_path: Path) -> None:
    write_seeds(tmp_path / "a.json")
    write_seeds(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert seedsplit.load(tmp_path / "a.json") == seedsplit.load(tmp_path / "b.json")
