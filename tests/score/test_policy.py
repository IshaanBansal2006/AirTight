from __future__ import annotations

import dataclasses
import itertools
import json
from collections import Counter
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
import pytest

from airtight.contracts import FleetConfig
from airtight.score import policy as pol
from airtight.score.policy import (
    FLOAT_FIELDS,
    Policy,
    chargers,
    named_policy,
    perturb,
    sample_docks,
    sample_policy,
    seed_policies,
)
from airtight.sim import scenarios
from airtight.sim.episode import (
    EpisodeParams,
    official_params,
    params_from_mapping,
    simulate,
    simulate_quiet,
)

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.contracts import Site

SITE = scenarios.load_site()  # two docks, capacity one each
CURVES = scenarios.load_sensor_curves()
GAINS = ("asset_gain", "entry_gain", "band_gain")
RANGES = {
    "asset_gain": pol.GAIN_RANGE,
    "entry_gain": pol.GAIN_RANGE,
    "band_gain": pol.BAND_RANGE,
    "weight_scale_m": pol.SCALE_RANGE_M,
    "d0_m": pol.D0_RANGE_M,
    "retarget_period_s": pol.RETARGET_RANGE_S,
    "top_fraction": pol.TOP_FRACTION_RANGE,
}
POLICY_PARAM_FIELDS = {*FLOAT_FIELDS, "weight_mode", "dock_assignment"}


def _rng() -> np.random.Generator:
    return np.random.default_rng([1, 2, 3])


def _example_fleet() -> FleetConfig:
    text = resources.files("airtight.contracts.examples").joinpath("fleet_config.json").read_text()
    return FleetConfig.model_validate_json(text)  # two drones, a Go2 and a guard


def _a_policy(**changes: object) -> Policy:
    base = Policy(
        asset_gain=1.5,
        entry_gain=0.0,
        band_gain=0.7,
        weight_scale_m=40.0,
        d0_m=80.0,
        retarget_period_s=6.0,
        top_fraction=0.1,
        offsets=(("d0", 0.0), ("d1", 1234.5)),
        docks=(("d0", "dock_ne"), ("d1", "dock_sw")),
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def _check_in_ranges(policy: Policy, fleet: FleetConfig, site: Site) -> None:
    for name in FLOAT_FIELDS:
        value = getattr(policy, name)
        lo, hi = RANGES[name]
        assert (name in GAINS and value == 0.0) or lo <= value <= hi, (name, value)
    cycles = dict(chargers(fleet))
    assert [agent for agent, _ in policy.offsets] == sorted(cycles)
    for agent, value in policy.offsets:
        assert 0.0 <= value <= cycles[agent], (agent, value)
    assert list(policy.docks) == sorted(policy.docks)
    assert {agent for agent, _ in policy.docks} <= set(cycles)
    assert {dock for _, dock in policy.docks} <= {d.id for d in site.docks}


def test_ranges_in_this_file_are_the_modules() -> None:
    assert RANGES == pol._RANGES
    assert set(FLOAT_FIELDS) == set(RANGES)
    assert GAINS == pol._GAINS


def test_chargers_lists_the_agents_that_charge_with_their_cycle() -> None:
    assert chargers(scenarios.load_fleet("2drones")) == [("d0", 3600.0), ("d1", 3600.0)]
    example = _example_fleet()
    got = dict(chargers(example))
    assert set(got) == {a.id for a in example.agents if a.charge_time_s > 0}
    assert all(got[a.id] == a.endurance_s + a.charge_time_s for a in example.agents if a.id in got)
    assert len(got) < len(example.agents)  # the guard never charges


def test_apply_changes_only_the_name_and_the_offsets() -> None:
    for fleet in (scenarios.load_fleet("2drones"), _example_fleet()):
        ids = [agent for agent, _ in chargers(fleet)]
        policy = _a_policy(offsets=tuple((a, 100.0 * i) for i, a in enumerate(sorted(ids))))
        new = policy.apply(fleet)
        assert new.agents == fleet.agents
        assert new.cost_per_hour() == fleet.cost_per_hour()
        assert new.cost_per_hour_by_type == fleet.cost_per_hour_by_type
        assert new.comms_mode == fleet.comms_mode
        assert new.charge_policy.threshold_frac == fleet.charge_policy.threshold_frac
        assert new.name == f"{fleet.name}__p{policy.digest()}" != fleet.name
        # a zero offset is left out, which is how the contract spells "synchronized"
        assert new.charge_policy.stagger_offsets_s == {
            a: 100.0 * i for i, a in enumerate(sorted(ids)) if i > 0
        }
        assert FleetConfig.model_validate(new.model_dump()) == new
        assert FleetConfig.model_validate_json(new.model_dump_json()) == new
        assert policy.apply(fleet, name="chosen").name == "chosen"
        changed = {k for k, v in new.model_dump().items() if v != fleet.model_dump()[k]}
        assert changed == {"name", "charge_policy"}
        assert fleet.charge_policy.stagger_offsets_s != new.charge_policy.stagger_offsets_s


def test_apply_replaces_offsets_the_fleet_already_had_and_refuses_strangers() -> None:
    staggered = scenarios.load_fleet("2drones_staggered")
    assert staggered.charge_policy.stagger_offsets_s  # the fixture really is staggered
    flat = named_policy(staggered, "asset").apply(staggered)
    assert flat.charge_policy.stagger_offsets_s == {}
    assert staggered.charge_policy.stagger_offsets_s  # and the input is left alone
    with pytest.raises(ValueError, match="unknown agents"):
        _a_policy(offsets=(("nobody", 5.0),)).apply(staggered)


def test_params_are_official_params_with_only_the_policy_fields_changed() -> None:
    policy = _a_policy()
    params, base = policy.params(), official_params()
    assert params.battery is True and base.battery is True
    assert params.weight_mode == "mix"
    assert params.dock_assignment == policy.docks
    for name in FLOAT_FIELDS:
        assert getattr(params, name) == getattr(policy, name)
    for f in dataclasses.fields(EpisodeParams):
        if f.name not in POLICY_PARAM_FIELDS:
            assert getattr(params, f.name) == getattr(base, f.name), f.name
    assert params.weight_base == base.weight_base  # the base stays; only gain ratios matter
    assert hash(params) == hash(policy.params())


def test_params_json_round_trips_to_the_same_params() -> None:
    fleet = scenarios.load_fleet("2drones")
    rng = _rng()
    candidates = [_a_policy(), _a_policy(docks=()), *seed_policies(fleet)]
    candidates += [sample_policy(SITE, fleet, rng) for _ in range(20)]
    for policy in candidates:
        mapping = policy.params_json()
        assert "battery" not in mapping and "task_time_s" not in mapping
        assert mapping["weight_mode"] == "mix"
        assert set(mapping) <= POLICY_PARAM_FIELDS
        assert params_from_mapping(mapping, official_params()) == policy.params()
    assert _a_policy().params_json()["dock_assignment"] == {"d0": "dock_ne", "d1": "dock_sw"}
    assert "dock_assignment" not in _a_policy(docks=()).params_json()


def test_params_json_reaches_official_params_through_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _a_policy()
    expected = policy.params()
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.params_json()))
    monkeypatch.setenv("AIRTIGHT_PARAMS_JSON", str(path))
    assert official_params() == expected


@pytest.mark.parametrize("mode", ["asset", "uniform", "band"])
def test_named_policies_simulate_bit_identically_to_the_named_modes(mode: str) -> None:
    fleet = scenarios.load_fleet("2drones")
    policy = named_policy(fleet, mode)
    plain = dataclasses.replace(official_params(), weight_mode=mode)
    if mode == "asset":
        assert plain == official_params()
    assert policy.params() != plain  # it is the mixture, not the named mode
    applied = policy.apply(fleet)
    assert applied.charge_policy.stagger_offsets_s == {}
    tactic = scenarios.load_tactic("jog").model_copy(update={"phase": 0.2})
    seen = 0
    for seed in (1000, 1001, 1002):
        want = simulate(SITE, fleet, tactic, CURVES, seed, plain)
        assert simulate(SITE, applied, tactic, CURVES, seed, policy.params()) == want
        assert simulate(SITE, fleet, tactic, CURVES, seed, policy.params()) == want
        seen += want.n_looks
    assert seen > 0  # the toy episodes do exercise the patrol and the sensors
    night = simulate_quiet(SITE, fleet, CURVES, 2000, params=plain)
    assert simulate_quiet(SITE, applied, CURVES, 2000, params=policy.params()) == night


def test_a_full_stagger_named_policy_is_the_staggered_fleet() -> None:
    fleet = scenarios.load_fleet("2drones")
    policy = named_policy(fleet, "asset", stagger_fraction=1.0)
    assert policy.offsets == (("d0", 0.0), ("d1", 1800.0))
    assert named_policy(fleet, "asset", 0.5).offsets == (("d0", 0.0), ("d1", 900.0))
    applied = policy.apply(fleet)
    assert applied.charge_policy.stagger_offsets_s == {"d1": 1800.0}
    by_hand = fleet.model_copy(
        update={
            "charge_policy": fleet.charge_policy.model_copy(
                update={"stagger_offsets_s": {"d1": 1800.0}}
            )
        }
    )
    tactic = scenarios.load_tactic("jog")
    for seed in (1000, 1001):
        want = simulate(SITE, by_hand, tactic, CURVES, seed, official_params())
        assert simulate(SITE, applied, tactic, CURVES, seed, policy.params()) == want


def test_named_policy_offsets_origin_and_unknown_modes() -> None:
    example = _example_fleet()
    full = named_policy(example, "band", 1.0, go2_half_cycle=True)
    assert dict(full.offsets) == {"drone_1": 0.0, "drone_2": 1950.0, "go2_1": 4500.0}
    assert full.apply(example).charge_policy.stagger_offsets_s == {
        "drone_2": 1950.0,
        "go2_1": 4500.0,
    }
    assert dict(named_policy(example, "band", 1.0).offsets)["go2_1"] == 0.0
    assert (full.asset_gain, full.entry_gain, full.band_gain) == (0.0, 0.0, 1.0)
    assert full.docks == ()
    assert full.origin == "named:band:s1:ghalf"
    assert named_policy(example, "asset").origin == "named:asset:s0:g0"
    assert named_policy(example, "asset", origin="mine").origin == "mine"
    base = official_params()
    for mode in ("asset", "uniform", "band"):
        p = named_policy(example, mode, 0.5)
        assert (p.weight_scale_m, p.d0_m, p.retarget_period_s, p.top_fraction) == (
            base.weight_scale_m,
            base.d0_m,
            base.retarget_period_s,
            base.top_fraction,
        )
    with pytest.raises(KeyError):
        named_policy(example, "mix")


def test_seed_policies_have_no_duplicate_digests() -> None:
    example = _example_fleet()
    seeds = seed_policies(example)
    assert len(seeds) == len({p.digest() for p in seeds}) == 3 * 2 * 2
    # without a Go2 the half-cycle option changes nothing; with one drone neither does stagger
    two = seed_policies(scenarios.load_fleet("2drones"))
    assert len(two) == len({p.digest() for p in two}) == 3 * 2
    one = seed_policies(scenarios.load_fleet("1drone"))
    assert len(one) == len({p.digest() for p in one}) == 3
    assert [p.digest() for p in seed_policies(example)] == [p.digest() for p in seeds]
    assert named_policy(example, "asset").digest() in {p.digest() for p in seeds}
    for p in seeds:
        _check_offsets_only(p, example)


def _check_offsets_only(policy: Policy, fleet: FleetConfig) -> None:
    cycles = dict(chargers(fleet))
    assert [agent for agent, _ in policy.offsets] == sorted(cycles)
    assert all(0.0 <= value <= cycles[agent] for agent, value in policy.offsets)


def test_digest_ignores_origin_and_nothing_else() -> None:
    policy = _a_policy()
    assert policy.digest() == _a_policy(origin="somewhere else").digest()
    assert len(policy.digest()) == 10
    changes: dict[str, object] = {name: getattr(policy, name) * 1.01 for name in FLOAT_FIELDS}
    changes["entry_gain"] = 0.3
    changes["offsets"] = (("d0", 0.0), ("d1", 1234.6))
    changes["docks"] = ()
    digests = {_a_policy(**{k: v}).digest() for k, v in changes.items()}
    assert len(digests) == len(changes) and policy.digest() not in digests
    assert "origin" in policy.describe()
    assert policy.describe()["offsets"] == {"d0": 0.0, "d1": 1234.5}
    assert policy.describe()["docks"] == {"d0": "dock_ne", "d1": "dock_sw"}


@pytest.mark.parametrize("fleet_name", ["1drone", "2drones", "4drones", "example"])
def test_sampling_is_deterministic_and_stays_inside_the_ranges(fleet_name: str) -> None:
    fleet = _example_fleet() if fleet_name == "example" else scenarios.load_fleet(fleet_name)
    first_rng, second_rng = _rng(), _rng()
    first = [sample_policy(SITE, fleet, first_rng) for _ in range(150)]
    second = [sample_policy(SITE, fleet, second_rng) for _ in range(150)]
    assert first == second
    assert len({p.digest() for p in first}) == len(first)
    for policy in first:
        _check_in_ranges(policy, fleet, SITE)
        assert policy.origin == "random"
        assert policy.params().weight_mode == "mix"
    for name in GAINS:  # both branches are taken: a gain is sometimes off, mostly on
        zeros = sum(getattr(p, name) == 0.0 for p in first)
        assert 0 < zeros < len(first) / 2
    assert 0 < sum(p.docks == () for p in first) < len(first)
    assert any(p.docks for p in first)
    other = sample_policy(SITE, fleet, np.random.default_rng([1, 2, 4]))
    assert other != first[0]


def test_sampled_policies_apply_and_simulate() -> None:
    fleet = scenarios.load_fleet("2drones")
    rng = _rng()
    tactic = scenarios.load_tactic("sprint")
    for _ in range(3):
        policy = sample_policy(SITE, fleet, rng)
        applied = policy.apply(fleet)
        assert applied.agents == fleet.agents
        scores = simulate(SITE, applied, tactic, CURVES, 1000, policy.params())
        assert scores == simulate(SITE, applied, tactic, CURVES, 1000, policy.params())
        near = perturb(SITE, fleet, policy, rng)
        simulate(SITE, near.apply(fleet), tactic, CURVES, 1000, near.params())


@pytest.mark.parametrize("fleet_name", ["2drones", "example"])
def test_perturbing_is_deterministic_and_stays_inside_the_ranges(fleet_name: str) -> None:
    fleet = _example_fleet() if fleet_name == "example" else scenarios.load_fleet(fleet_name)

    def chains(rng: np.random.Generator) -> list[list[Policy]]:
        out = []
        for start in (named_policy(fleet, "uniform"), sample_policy(SITE, fleet, rng)):
            chain = [start]
            for _ in range(150):
                chain.append(perturb(SITE, fleet, chain[-1], rng))
            out.append(chain)
        return out

    first = chains(_rng())
    assert first == chains(_rng())
    switched_on = switched_off = moved = 0
    for chain in first:
        for parent, child in itertools.pairwise(chain):
            _check_in_ranges(child, fleet, SITE)
            assert child.origin == f"near:{parent.digest()}"
            for name in GAINS:
                switched_on += getattr(parent, name) == 0.0 and getattr(child, name) > 0.0
                switched_off += getattr(parent, name) > 0.0 and getattr(child, name) == 0.0
            differing = set(child.docks) ^ set(parent.docks)
            assert len({agent for agent, _ in differing}) <= 1  # at most one dock moves
            moved += bool(differing)
            for name in set(FLOAT_FIELDS) - set(GAINS):  # a nudge, not a jump
                ratio = getattr(child, name) / getattr(parent, name)
                assert abs(np.log(ratio)) <= 6 * pol.PERTURB_SIGMA
    assert switched_on > 0 and switched_off > 0 and moved > 0


def test_perturb_keeps_a_policy_without_chargers_or_docks_well_formed() -> None:
    fleet = scenarios.load_fleet("2drones")
    no_charge = fleet.model_copy(
        update={"agents": [a.model_copy(update={"charge_time_s": 0.0}) for a in fleet.agents]}
    )
    no_docks = SITE.model_copy(update={"docks": []})
    rng = _rng()
    for site, flt in [(SITE, no_charge), (no_docks, fleet), (no_docks, no_charge)]:
        for _ in range(30):
            policy = sample_policy(site, flt, rng)
            near = perturb(site, flt, policy, rng)
            for p in (policy, near):
                _check_in_ranges(p, flt, site)
                if not site.docks or not chargers(flt):
                    assert p.docks == ()
            if not chargers(flt):
                assert policy.offsets == near.offsets == ()
                assert policy.apply(flt).charge_policy.stagger_offsets_s == {}


def test_sample_docks_respects_capacity_when_the_chargers_fit() -> None:
    rng = _rng()
    dock_ids = {d.id for d in SITE.docks}
    capacity = {d.id: int(d.capacity) for d in SITE.docks}
    assert sum(capacity.values()) == 2
    fits = scenarios.load_fleet("2drones")
    seen = set()
    for _ in range(40):
        docks = sample_docks(SITE, fits, rng)
        assert [agent for agent, _ in docks] == ["d0", "d1"]
        used = Counter(dock for _, dock in docks)
        assert set(used) <= dock_ids
        assert all(used[d] <= capacity[d] for d in used)
        seen.add(docks)
    assert len(seen) == 2  # both ways round turn up
    roomy = SITE.model_copy(
        update={"docks": [d.model_copy(update={"capacity": 3}) for d in SITE.docks]}
    )
    four = scenarios.load_fleet("4drones")
    for _ in range(40):
        used = Counter(dock for _, dock in sample_docks(roomy, four, rng))
        assert sum(used.values()) == 4 and all(n <= 3 for n in used.values())


def test_sample_docks_spreads_the_overflow_and_handles_the_empty_cases() -> None:
    rng = _rng()
    four = scenarios.load_fleet("4drones")
    assert len(chargers(four)) == 4 > sum(int(d.capacity) for d in SITE.docks)
    for _ in range(40):
        docks = sample_docks(SITE, four, rng)
        assert [agent for agent, _ in docks] == sorted(a.id for a in four.agents)
        used = Counter(dock for _, dock in docks)
        assert set(used) <= {d.id for d in SITE.docks}
        assert sorted(used.values()) == [2, 2]  # evenly, when capacity cannot be kept
    assert sample_docks(SITE.model_copy(update={"docks": []}), four, rng) == ()
    no_charge = four.model_copy(
        update={"agents": [a.model_copy(update={"charge_time_s": 0.0}) for a in four.agents]}
    )
    assert sample_docks(SITE, no_charge, rng) == ()
    first = sample_docks(SITE, four, _rng())
    assert first == sample_docks(SITE, four, _rng())
