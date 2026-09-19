from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

from airtight.contracts import XY, BenignRoute, Decoy, Site, Tactic
from airtight.sim import adapt
from airtight.sim.actors import (
    BenignObject,
    Intruder,
    SimObject,
    benign_rng,
    make_decoy,
    spawn_benign,
)
from airtight.sim.constants import DECOY_DURATION_S

EXAMPLES = resources.files("airtight.contracts.examples")
TACTIC_FILES = sorted(
    f.name for f in EXAMPLES.iterdir() if f.name.startswith("tactic") and f.name.endswith(".json")
)


def _route(route_id: str, x0: float, x1: float, rate: float, cls: str = "person") -> BenignRoute:
    return BenignRoute(
        id=route_id,
        cls=cls,
        waypoints=[XY(x=x0, y=10.0), XY(x=x1, y=10.0)],
        arrival_rate_per_hour=rate,
    )


@pytest.fixture
def site() -> Site:
    return Site.model_validate_json(EXAMPLES.joinpath("site.json").read_text())


@pytest.fixture
def tactic() -> Tactic:
    return Tactic.model_validate_json(EXAMPLES.joinpath("tactic.json").read_text())


def _with_routes(site: Site, *routes: BenignRoute) -> Site:
    return site.model_copy(update={"benign_routes": list(routes)})


def _summary(objects: list[BenignObject]) -> list[tuple[str, str, float]]:
    return [(o.object_id, o.kind, o.t_start) for o in objects]


@pytest.mark.parametrize("filename", TACTIC_FILES)
def test_intruder_timeline(site: Site, filename: str) -> None:
    tactic = Tactic.model_validate_json(EXAMPLES.joinpath(filename).read_text())
    intruder = Intruder(site, tactic)
    entry = site.entry(tactic.entry_id).position
    assert (intruder.object_id, intruder.kind) == ("intruder", "intruder")
    assert intruder.t_reach == adapt.t_reach(site, tactic)
    assert intruder.t_cdp == adapt.t_cdp(site, tactic)
    assert not intruder.alive(-0.25) and intruder.alive(0.0) and intruder.alive(1e6)
    assert intruder.position(0.0).tolist() == [entry.x, entry.y]
    last = tactic.waypoints[-1]
    for t in (intruder.t_reach, intruder.t_reach + 0.25, intruder.t_reach + 1e4):
        assert intruder.position(t).tolist() == [last.x, last.y]
    just_before = intruder.position(intruder.t_reach - 1.0)
    assert np.linalg.norm(just_before - [last.x, last.y]) == pytest.approx(tactic.speed_mps)


def test_example_intruder_ends_on_the_asset(site: Site, tactic: Tactic) -> None:
    intruder = Intruder(site, tactic)
    assert intruder.position(intruder.t_reach).tolist() == adapt.assets(site)[0].tolist()


def test_spawn_benign_same_seed_same_output_other_seed_differs(site: Site) -> None:
    busy = _with_routes(site, _route("walk", 10.0, 110.0, 200.0))
    first, again = spawn_benign(busy, 0.0, 200.0, 7), spawn_benign(busy, 0.0, 200.0, 7)
    assert len(first) > 0
    assert _summary(first) == _summary(again)
    assert _summary(first) != _summary(spawn_benign(busy, 0.0, 200.0, 8))


def test_adding_a_route_leaves_the_first_routes_objects_identical(site: Site) -> None:
    walk = _route("walk", 10.0, 110.0, 200.0)
    alone = spawn_benign(_with_routes(site, walk), 0.0, 200.0, 7)
    # "cart" sorts before "walk" and is listed first, so order cannot be what protects "walk"
    both = spawn_benign(_with_routes(site, _route("cart", 20.0, 90.0, 300.0), walk), 0.0, 200.0, 7)
    assert any(o.object_id.startswith("cart-") for o in both)
    assert _summary(alone) == _summary([o for o in both if o.object_id.startswith("walk-")])


def test_output_is_sorted_with_ids_in_start_time_order(site: Site) -> None:
    routes = (_route("walk", 10.0, 110.0, 200.0), _route("cart", 20.0, 90.0, 300.0))
    objects = spawn_benign(_with_routes(site, *routes), 0.0, 200.0, 7)
    keys = [(o.object_id.rsplit("-", 1)[0], o.t_start) for o in objects]
    assert keys == sorted(keys)
    walkers = [o for o in objects if o.object_id.startswith("walk-")]
    assert [o.object_id for o in walkers] == [f"walk-{i}" for i in range(len(walkers))]


def test_mean_count_matches_the_poisson_rate(site: Site) -> None:
    t0, t1 = 0.0, 200.0
    window = (t1 - t0) + 100.0 / adapt.benign_speed("person")  # route duration starts it early
    rate_per_hour = 8.0 * 3600.0 / window
    one = _with_routes(site, _route("walk", 10.0, 110.0, rate_per_hour))
    counts = [len(spawn_benign(one, t0, t1, seed)) for seed in range(200)]
    expected = rate_per_hour / 3600.0 * window
    assert expected == pytest.approx(8.0)
    assert abs(np.mean(counts) - expected) < 0.10 * expected


def test_some_object_is_already_partway_along_at_t0(site: Site) -> None:
    busy = _with_routes(site, _route("walk", 10.0, 110.0, 600.0))
    t0 = 0.0
    midway = [
        o
        for o in spawn_benign(busy, t0, 60.0, 3)
        if o.alive(t0) and 10.0 < float(o.position(t0)[0]) < 110.0
    ]
    assert midway and all(o.t_start < t0 for o in midway)


def test_every_spawned_object_overlaps_the_window(site: Site) -> None:
    busy = _with_routes(site, _route("walk", 10.0, 110.0, 600.0))
    objects = spawn_benign(busy, 50.0, 120.0, 3)
    assert objects and all(o.t_end >= 50.0 and o.t_start <= 120.0 for o in objects)


def test_zero_rate_route_yields_nothing_and_no_routes_yields_nothing(site: Site) -> None:
    assert spawn_benign(_with_routes(site, _route("quiet", 10.0, 110.0, 0.0)), 0.0, 3600.0, 1) == []
    assert spawn_benign(_with_routes(site), 0.0, 3600.0, 1) == []
    with pytest.raises(ValueError, match="t0 <= t1"):
        spawn_benign(site, 10.0, 0.0, 1)


def test_benign_rng_depends_on_seed_and_route_only() -> None:
    assert benign_rng(7, "walk").bit_generator.state == benign_rng(7, "walk").bit_generator.state
    assert benign_rng(7, "walk").bit_generator.state != benign_rng(7, "cart").bit_generator.state
    assert benign_rng(7, "walk").bit_generator.state != benign_rng(8, "walk").bit_generator.state


def test_benign_object_lifetime_and_position() -> None:
    points = np.array([[10.0, 10.0], [10.0, 30.0], [40.0, 30.0]])  # 50 m
    obj = BenignObject("fox-0", "animal", points, speed_mps=2.0, t_start=-5.0)
    assert (obj.object_id, obj.kind, obj.t_end) == ("fox-0", "animal", 20.0)
    assert not obj.alive(-5.01) and obj.alive(-5.0) and obj.alive(20.0) and not obj.alive(20.01)
    assert obj.position(-5.0).tolist() == [10.0, 10.0]
    assert obj.position(5.0).tolist() == [10.0, 30.0]  # the corner, 20 m along
    assert obj.position(20.0).tolist() == [40.0, 30.0]


def test_make_decoy_none_without_a_decoy(tactic: Tactic) -> None:
    assert tactic.decoy is None
    assert make_decoy(tactic) is None


def test_decoy_is_alive_exactly_on_its_interval(tactic: Tactic) -> None:
    lure = tactic.model_copy(update={"decoy": Decoy(position=XY(x=100, y=60), lead_time_s=40)})
    decoy = make_decoy(lure)
    assert decoy is not None
    assert (decoy.object_id, decoy.kind) == ("decoy", "decoy")
    assert decoy.t_on == -40.0 and decoy.t_off == -40.0 + DECOY_DURATION_S
    assert not decoy.alive(-40.01) and decoy.alive(-40.0)
    assert decoy.alive(0.0) and decoy.alive(decoy.t_off) and not decoy.alive(decoy.t_off + 0.01)
    assert decoy.position(0.0).tolist() == [100.0, 60.0]


def test_position_always_returns_a_copy(site: Site, tactic: Tactic) -> None:
    lure = tactic.model_copy(update={"decoy": Decoy(position=XY(x=100, y=60), lead_time_s=40)})
    intruder, decoy = Intruder(site, lure), make_decoy(lure)
    benign = BenignObject("walk-0", "person", np.array([[0.0, 0.0], [10.0, 0.0]]), 1.0, 0.0)
    assert decoy is not None
    objects: list[SimObject] = [intruder, decoy, benign]
    for obj in objects:
        for t in (0.0, 3.0, 1e5):  # start, midway, clamped at the end
            before = obj.position(t).copy()
            obj.position(t)[:] = -999.0
            assert np.array_equal(obj.position(t), before)
