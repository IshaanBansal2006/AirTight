from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from airtight.contracts import XY, Site, Tactic
from airtight.score.adversary import (
    MAX_T_REACH_S,
    Limits,
    keep_most_harmful,
    key_hash,
    load_limits,
    load_tactic_files,
    random_tactics,
    strong_grid,
    t_reach_s,
    validate_tactic,
)
from airtight.sim import scenarios

LIMITS = Limits(
    speed_min_mps=0.8,
    speed_cap_mps=2.0,
    max_waypoints=6,
    perimeter_tol_m=1.5,
    asset_tol_m=1.0,
    min_leg_m=2.0,
)
REPO = Path(__file__).resolve().parents[2]
REAL_SCENARIO = REPO / "data" / "lane_c_export" / "scenarios" / "logistics_yard"
REAL_TACTICS = REPO / "data" / "lane_c_export" / "results" / "v3" / "tactics"


@pytest.fixture
def site() -> Site:
    return scenarios.load_site()


def _straight(site: Site, **changes: object) -> Tactic:
    base = Tactic(
        id="straight",
        family="charging_window",
        entry_id=site.entry_points[0].id,
        phase=0.25,
        speed_mps=LIMITS.speed_cap_mps,
        waypoints=[site.asset],
    )
    return base.model_copy(update=changes)


def _as_json(tactic: Tactic) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(tactic.model_dump_json())
    return loaded


def test_load_limits_reads_the_config_and_names_what_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="redteam_config.json"):
        load_limits(tmp_path)
    full = dataclasses.asdict(LIMITS) | {"decoy_offset_m": 8.0}
    (tmp_path / "redteam_config.json").write_text(json.dumps(full))
    assert load_limits(tmp_path) == LIMITS
    del full["min_leg_m"]
    (tmp_path / "redteam_config.json").write_text(json.dumps(full))
    with pytest.raises(ValueError, match="min_leg_m"):
        load_limits(tmp_path)
    (tmp_path / "redteam_config.json").write_text("not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_limits(tmp_path)


def test_strong_grid_counts_ids_and_shape(site: Site) -> None:
    grid = strong_grid(site, LIMITS, n_phases=12)
    assert len(grid) == len(site.entry_points) * 12 * 2
    assert len({t.id for t in grid}) == len(grid)
    assert {t.speed_mps for t in grid} == {LIMITS.speed_min_mps, LIMITS.speed_cap_mps}
    assert sorted({t.phase for t in grid}) == [k / 12 for k in range(12)]
    assert all(t.waypoints == [site.asset] for t in grid)
    assert all(t.family == "charging_window" and t.origin == "hand" for t in grid)
    first = grid[0]
    assert first.id == f"strong-{site.entry_points[0].id}-{first.speed_mps:g}-{first.phase:.4f}"
    assert len(strong_grid(site, LIMITS)) == len(site.entry_points) * 48 * 2
    assert strong_grid(site, LIMITS, 12) == grid


def test_strong_grid_skips_a_tactic_too_slow_to_fit_the_window(site: Site) -> None:
    crawl = dataclasses.replace(LIMITS, speed_min_mps=0.001)
    grid = strong_grid(site, crawl, n_phases=4)
    assert len(grid) == len(site.entry_points) * 4
    assert {t.speed_mps for t in grid} == {LIMITS.speed_cap_mps}


def test_generated_tactics_are_valid_and_survive_a_json_round_trip(site: Site) -> None:
    generated = [*strong_grid(site, LIMITS, 8), *random_tactics(site, LIMITS, 60, [7, 1])]
    for tactic in generated:
        assert validate_tactic(site, LIMITS, tactic) == []
        again = Tactic.model_validate_json(tactic.model_dump_json())
        assert again == tactic and again.content_hash() == tactic.content_hash()


def test_random_tactics_are_deterministic_and_keyed(site: Site) -> None:
    a = random_tactics(site, LIMITS, 40, [7, 1])
    assert a == random_tactics(site, LIMITS, 40, (7, 1))
    assert a[:10] == random_tactics(site, LIMITS, 10, [7, 1])
    b = random_tactics(site, LIMITS, 40, [7, 2])
    assert not {t.id for t in a} & {t.id for t in b}
    assert [t.waypoints for t in a] != [t.waypoints for t in b]
    assert [t.id for t in a] == [f"rand-{key_hash([7, 1])}-{i:04d}" for i in range(40)]
    assert random_tactics(site, LIMITS, 0, [7, 1]) == []


def test_random_tactics_respect_every_limit(site: Site) -> None:
    tactics = random_tactics(site, LIMITS, 200, [11])
    known = {e.id for e in site.entry_points}
    lengths = {len(t.waypoints) for t in tactics}
    speeds = [t.speed_mps for t in tactics]
    assert all(t.entry_id in known and 0.0 <= t.phase < 1.0 for t in tactics)
    assert all(t.waypoints[-1] == site.asset for t in tactics)
    assert min(lengths) >= 1 and max(lengths) <= LIMITS.max_waypoints and len(lengths) > 1
    assert all(LIMITS.speed_min_mps <= s <= LIMITS.speed_cap_mps for s in speeds)
    assert LIMITS.speed_min_mps in speeds and LIMITS.speed_cap_mps in speeds
    assert any(LIMITS.speed_min_mps < s < LIMITS.speed_cap_mps for s in speeds)
    assert all(t_reach_s(site, t) <= MAX_T_REACH_S for t in tactics)
    assert all(t.family == "blind_spot" and t.origin == "search" for t in tactics)
    for tactic in tactics:
        path = [site.entry(tactic.entry_id).position, *tactic.waypoints]
        legs = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:], strict=False)]
        assert min(legs) >= LIMITS.min_leg_m
        for p in tactic.waypoints[:-1]:
            assert p.x == round(p.x, 2) and p.y == round(p.y, 2)


def test_validate_tactic_names_each_violation(site: Site) -> None:
    assert validate_tactic(site, LIMITS, _straight(site)) == []
    xmin, ymin, xmax, ymax = site.bounds
    far = XY(x=xmax + 50.0, y=ymax + 50.0)
    near_asset = XY(x=site.asset.x + 0.5, y=site.asset.y)
    off_asset = XY(x=site.asset.x + 5.0, y=site.asset.y)
    cases: list[tuple[dict[str, object], str]] = [
        ({"entry_id": "nowhere"}, "unknown entry"),
        ({"speed_mps": LIMITS.speed_cap_mps + 0.01}, "speed"),
        ({"speed_mps": LIMITS.speed_min_mps - 0.01}, "speed"),
        ({"waypoints": [site.asset] * (LIMITS.max_waypoints + 1)}, "waypoints exceeds"),
        ({"waypoints": [off_asset]}, "from the asset"),
        ({"waypoints": [far, site.asset]}, "outside the perimeter"),
        ({"waypoints": [near_asset, site.asset]}, "leg shorter"),
        ({"family": "decoy"}, "requires a decoy"),
        ({"family": "comms_cut"}, "requires a comms event"),
    ]
    for changes, expected in cases:
        reasons = validate_tactic(site, LIMITS, _straight(site, **changes))
        assert any(expected in r for r in reasons), (changes, reasons)
    assert validate_tactic(site, LIMITS, _straight(site, waypoints=[near_asset])) == []
    edge = _straight(site, speed_mps=LIMITS.speed_cap_mps + 1e-12)
    assert validate_tactic(site, LIMITS, edge) == []
    crawl = dataclasses.replace(LIMITS, speed_min_mps=0.001)
    reasons = validate_tactic(site, crawl, _straight(site, speed_mps=0.001))
    assert len(reasons) == 1 and "t_reach" in reasons[0]


def test_loader_handles_bare_list_wrapped_and_garbage(site: Site, tmp_path: Path) -> None:
    bare = _straight(site, id="bare")
    listed = [_straight(site, id="listed-a", phase=0.1), _straight(site, id="listed-b", phase=0.2)]
    wrapped = _straight(site, id="wrapped", phase=0.3)
    deep = _straight(site, id="deep", phase=0.4)
    clash = _straight(site, id="bare", phase=0.9)
    too_fast = _straight(site, id="too-fast", speed_mps=LIMITS.speed_cap_mps * 3)
    root = tmp_path / "tactics"
    (root / "nested").mkdir(parents=True)
    (root / "a_bare.json").write_text(bare.model_dump_json())
    (root / "b_list.json").write_text(json.dumps([_as_json(t) for t in listed]))
    (root / "c_wrapped.json").write_text(
        json.dumps([{"score": 1.0, "tactic": _as_json(wrapped)}, {"tactic": _as_json(bare)}])
    )
    (root / "nested" / "d_deep.json").write_text(
        json.dumps({"family": "x", "scores": [{"tactic_id": "deep"}], "tactics": [_as_json(deep)]})
    )
    (root / "e_clash.json").write_text(clash.model_dump_json())
    (root / "f_bad.json").write_text(
        json.dumps([_as_json(too_fast), _as_json(bare) | {"phase": 2.0, "id": "bad-phase"}])
    )
    (root / "garbage.json").write_text("{ this is not json")
    (root / "summary.json").write_text(json.dumps({"site": "yard", "families": {"decoy": {}}}))
    (root / "ignored.txt").write_text("not a json file")

    tactics, notes = load_tactic_files([root, tmp_path / "missing"], site, LIMITS)
    ids = [t.id for t in tactics]
    assert len(ids) == len(set(ids)) == 6
    assert {"listed-a", "listed-b", "wrapped", "deep"} <= set(ids)
    assert f"bare-{bare.content_hash()[:6]}" in ids and f"bare-{clash.content_hash()[:6]}" in ids
    assert all(validate_tactic(site, LIMITS, t) == [] for t in tactics)
    text = "\n".join(notes)
    for expected in (
        "garbage.json",
        "summary.json: holds no tactic",
        "missing: no such file",
        "'too-fast': outside the limits: speed",
        "'bad-phase': fails the Tactic contract",
        "exact duplicate",
        "id 'bare' names 2 tactics",
    ):
        assert expected in text, (expected, notes)
    assert load_tactic_files([root, tmp_path / "missing"], site, LIMITS) == (tactics, notes)
    assert [t.id for t in load_tactic_files([root / "b_list.json"], site, LIMITS)[0]] == [
        "listed-a",
        "listed-b",
    ]


def test_keep_most_harmful_orders_by_detection_then_id(site: Site) -> None:
    tactics = [_straight(site, id=name) for name in ("d", "b", "c", "a", "e")]
    detection = {"a": 0.5, "b": 0.1, "c": 0.5, "d": 0.0, "e": 0.9, "unused": 0.0}
    assert [t.id for t in keep_most_harmful(tactics, detection, keep=4)] == ["d", "b", "a", "c"]
    assert [t.id for t in keep_most_harmful(tactics[::-1], detection, keep=4)] == [
        "d",
        "b",
        "a",
        "c",
    ]
    assert len(keep_most_harmful(tactics, detection)) == 5
    assert keep_most_harmful(tactics, detection, keep=0) == []
    with pytest.raises(ValueError, match="no measured detection"):
        keep_most_harmful(tactics, {"a": 0.5})


def test_real_export_generators_stay_inside_lane_cs_limits() -> None:
    if not (REAL_SCENARIO / "site.json").is_file():
        pytest.skip("lane C's logistics_yard export is not on this machine")
    real_site = Site.model_validate_json((REAL_SCENARIO / "site.json").read_text())
    limits = load_limits(REAL_SCENARIO)
    grid = strong_grid(real_site, limits)
    assert len(grid) == len(real_site.entry_points) * 48 * 2
    rand = random_tactics(real_site, limits, 100, [2026, 1])
    assert rand == random_tactics(real_site, limits, 100, [2026, 1])
    assert all(validate_tactic(real_site, limits, t) == [] for t in [*grid, *rand])


def test_real_lane_c_tactic_files_load() -> None:
    if not (REAL_SCENARIO / "site.json").is_file() or not REAL_TACTICS.is_dir():
        pytest.skip("lane C's tactic files are not on this machine")
    real_site = Site.model_validate_json((REAL_SCENARIO / "site.json").read_text())
    limits = load_limits(REAL_SCENARIO)
    tactics, notes = load_tactic_files([REAL_TACTICS], real_site, limits)
    assert tactics, notes
    assert len({t.id for t in tactics}) == len(tactics)
    assert all(validate_tactic(real_site, limits, t) == [] for t in tactics)
    assert any("summary.json" in note for note in notes)
