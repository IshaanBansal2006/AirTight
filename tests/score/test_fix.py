from __future__ import annotations

import dataclasses
import json
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from airtight.contracts import FleetConfig, read_episode_log
from airtight.score import fix
from airtight.score.fix import (
    Candidate,
    CandidateScore,
    choose_fix,
    confirm,
    export_pairs,
    find_pairs,
    make_candidates,
    own_gap_tactics,
    run_search,
)
from airtight.score.report import METRIC_DECISIONS, METRIC_PD, METRIC_WORST, build_report
from airtight.score.sweep import fleet_gaps, load_inputs, run_sweep
from airtight.sim import episode, scenarios
from airtight.sim.constants import NEVER_SEEN
from airtight.sim.episode import EpisodeScores

if TYPE_CHECKING:
    from airtight.score.sweep import SweepInputs

SEEDS = list(range(1000, 1010))
QUIET_SEEDS = [2000, 2001]


def _example_fleet() -> FleetConfig:
    text = resources.files("airtight.contracts.examples").joinpath("fleet_config.json").read_text()
    return FleetConfig.model_validate_json(text)  # two drones, a Go2 and a guard


def test_thirty_candidates_with_exactly_the_baselines_agents_and_cost() -> None:
    baseline = _example_fleet()
    candidates = make_candidates(baseline)
    assert len(candidates) == 5 * 3 * 2 == 30
    assert len({c.name for c in candidates}) == 30
    for c in candidates:
        assert c.fleet.agents == baseline.agents  # same agents, field for field
        assert c.fleet.cost_per_hour() == baseline.cost_per_hour()
        assert c.fleet.cost_per_hour_by_type == baseline.cost_per_hour_by_type
        assert c.fleet.comms_mode == baseline.comms_mode
        assert c.fleet.charge_policy.threshold_frac == baseline.charge_policy.threshold_frac
        assert c.params() == dataclasses.replace(
            episode.official_params(), weight_mode=c.weight_mode
        )
    first = candidates[0]  # no stagger, asset mode, Go2 offset 0: the baseline itself
    assert (first.stagger_fraction, first.weight_mode, first.go2_half_cycle, first.change) == (
        0,
        "asset",
        False,
        0,
    )
    assert first.fleet.charge_policy.stagger_offsets_s == {}
    full = next(c for c in candidates if c.stagger_fraction == 1.0 and c.go2_half_cycle)
    # drones: even spacing over their 3900 s cycle; the Go2: half of its own 9000 s cycle
    assert full.fleet.charge_policy.stagger_offsets_s == {"drone_2": 1950.0, "go2_1": 4500.0}
    half = next(c for c in candidates if c.stagger_fraction == 0.5 and not c.go2_half_cycle)
    assert half.fleet.charge_policy.stagger_offsets_s == {"drone_2": 975.0}


def test_a_fleet_without_a_go2_has_no_go2_option() -> None:
    assert len(make_candidates(scenarios.load_fleet("2drones"))) == 5 * 3


def _row(
    worst: float, pd: float, fraction: float, mode: str = "asset", go2: bool = False
) -> CandidateScore:
    cand = Candidate(
        f"c{fraction}{mode}{go2}", scenarios.load_fleet("2drones"), mode, fraction, go2
    )
    return CandidateScore(cand, 0.0, 5, 2.0, "ok", pd, "t", worst)


def test_choose_fix_order_of_preference() -> None:
    assert choose_fix([_row(0.1, 0.9, 0.0), _row(0.3, 0.2, 1.0)]).candidate.stagger_fraction == 1.0
    assert (
        choose_fix([_row(0.3, 0.5, 0.25), _row(0.3, 0.6, 1.0)]).pd == 0.6
    )  # tie: overall detection
    tie = [_row(0.3, 0.6, 1.0, "band"), _row(0.3, 0.6, 0.5), _row(0.3, 0.6, 0.5, go2=True)]
    assert choose_fix(tie) is tie[1]  # tie again: the smallest change from the baseline
    with pytest.raises(ValueError, match="at least one"):
        choose_fix([])


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two drones on duty 2000 s of a 3600 s cycle. Synchronized leaves a 1600 s gap, a
    half-spacing stagger a 700 s one, and full even spacing none at all."""
    tmp = tmp_path_factory.mktemp("fix")
    root = tmp / "scenario"
    base = scenarios.load_fleet("2drones")
    drone = base.agents[0].model_copy(update={"endurance_s": 2000.0, "charge_time_s": 1600.0})
    agents = [drone.model_copy(update={"id": f"d{i}"}) for i in range(2)]
    fleet = base.model_copy(update={"name": "toy_sync", "agents": agents})
    for name, model in (
        ("site.json", scenarios.load_site()),
        ("sensor_curve.json", scenarios.load_sensor_curves()),
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(model.model_dump_json())
    (root / "fleets").mkdir()
    (root / "fleets" / "toy_sync.json").write_text(fleet.model_dump_json())
    (root / "fleets" / "sweep.json").write_text(
        json.dumps({"baseline": "toy_sync", "configs": ["toy_sync"]})
    )
    (root / "redteam_config.json").write_text(
        json.dumps({"speed_cap_mps": 1.4})
    )  # a walker: easy to see
    inputs = load_inputs(root)
    out = tmp / "sweep"
    result = run_sweep(inputs, SEEDS, QUIET_SEEDS, out, workers=1)
    _, detail = build_report(result, n_boot=50)
    candidates = make_candidates(fleet, fractions=(0.0, 0.5, 1.0), modes=("asset",))
    rows = run_search(
        inputs,
        detail.configs["toy_sync"].pd_by_tactic,
        candidates,
        SEEDS,
        QUIET_SEEDS,
        out,
        workers=1,
        n_worst=3,
        n_grid=3,
        n_boot=50,
    )
    winner = choose_fix(rows)
    conf = confirm(
        inputs, winner.candidate, "toy_sync_fixed", SEEDS, QUIET_SEEDS, out, workers=1, n_boot=200
    )
    return {
        "tmp": tmp,
        "inputs": inputs,
        "result": result,
        "rows": rows,
        "winner": winner,
        "conf": conf,
    }


def test_each_candidate_is_attacked_inside_its_own_gaps(toy: dict[str, Any]) -> None:
    inputs: SweepInputs = toy["inputs"]
    by_fraction = {r.candidate.stagger_fraction: r for r in toy["rows"]}
    assert [round(r.drones_down_s_per_hour) for r in toy["rows"]] == [1600, 700, 0]
    for fraction, row in by_fraction.items():
        own = own_gap_tactics(inputs, row.candidate.fleet, [])
        gaps = fleet_gaps(row.candidate.fleet)["drones_down"]
        assert len(own) == len(gaps) * len(inputs.site.entry_points)
        assert all(any(a <= t.phase < b for a, b in gaps) for t in own)
        assert all(t.speed_mps == 1.4 and t.id.startswith("own-") for t in own)
        # against a set that already attacks those gaps, nothing is added twice
        attacked = [*inputs.tactics, *own]
        assert own_gap_tactics(inputs, row.candidate.fleet, attacked) == []
        if gaps:  # nobody is up inside its own gap, so that is where it is worst
            assert row.worst_tactic_pd == 0.0, fraction


def test_the_search_returns_the_stagger_that_closes_the_gap(toy: dict[str, Any]) -> None:
    winner: CandidateScore = toy["winner"]
    assert winner.candidate.stagger_fraction == 1.0 and winner.candidate.weight_mode == "asset"
    assert winner.drones_down_s_per_hour == 0.0 and winner.worst_tactic_pd > 0.0
    assert all(r.worst_tactic_pd == 0.0 for r in toy["rows"] if r is not winner)
    table = fix.format_search(toy["rows"], winner)
    assert table.count("<- winner") == 1 and len(table.splitlines()) == 1 + 3
    assert [r.name for r in fix.search_rows(toy["rows"])] == [r.candidate.name for r in toy["rows"]]


def test_confirmation_is_paired_on_the_same_tactics_and_seeds(toy: dict[str, Any]) -> None:
    conf = toy["conf"]
    base_eps, fix_eps = conf.baseline.episodes["toy_sync"], conf.fixed.episodes["toy_sync_fixed"]
    assert list(base_eps) == list(fix_eps) == [t.id for t in conf.tactics]  # the same tactic set
    assert all(
        [e.seed for e in base_eps[t]] == SEEDS == [e.seed for e in fix_eps[t]] for t in base_eps
    )
    assert len(conf.tactics) >= len(toy["inputs"].tactics)  # the full set, plus the fix's own gaps
    assert [d.metric for d in conf.deltas] == [METRIC_PD, METRIC_WORST, METRIC_DECISIONS]
    for d in conf.deltas:
        assert d.ci[0] <= d.ci[1]
    by = {d.metric: d for d in conf.deltas}
    assert by[METRIC_WORST].delta == pytest.approx(
        conf.fixed_score.worst_tactic_pd - conf.baseline_score.worst_tactic_pd
    )
    assert conf.baseline_score.worst_tactic_pd == 0.0 and conf.fixed_score.worst_tactic_pd > 0.0
    # the baseline's worst is exactly 0, so no replicate can put the fix below it
    assert conf.is_fix and by[METRIC_WORST].delta > 0
    assert by[METRIC_WORST].ci[0] >= 0 and by[METRIC_WORST].ci[1] > 0


def test_a_winner_that_does_not_beat_the_baseline_is_not_called_a_fix(toy: dict[str, Any]) -> None:
    inputs: SweepInputs = toy["inputs"]
    same = make_candidates(inputs.fleets["toy_sync"], fractions=(0.0,), modes=("asset",))[0]
    conf = confirm(
        inputs,
        same,
        "toy_sync_same",
        SEEDS,
        QUIET_SEEDS,
        toy["tmp"] / "sweep",
        workers=1,
        n_boot=50,
    )
    assert not conf.is_fix
    assert all(d.delta == 0.0 and d.ci == (0.0, 0.0) for d in conf.deltas)  # itself: exactly zero
    detail = fix.fix_detail(inputs, same, "toy_sync_same", conf, toy["rows"], 10, 2)
    assert detail.is_fix is False and detail.verdict.startswith("NOT a fix")
    report, base_detail = build_report(toy["result"], n_boot=50)
    new_report, new_detail = fix.add_fix_to_report(
        report, base_detail, inputs, same, "toy_sync_same", detail
    )
    assert [c.config_name for c in new_report.configs] == [
        "toy_sync"
    ]  # nothing added to the contract
    assert new_detail.fix is not None and new_detail.fix.is_fix is False
    assert new_detail.fix.status == "in-sample only" and len(new_detail.fix.search) == 3


HELD_OUT_SEEDS = list(range(3000, 3012))
HELD_OUT_QUIET = [4000, 4001]


@pytest.fixture(scope="module")
def held(toy: dict[str, Any]) -> fix.HeldOut:
    return fix.held_out_confirmation(
        toy["inputs"],
        toy["winner"].candidate,
        toy["conf"],
        HELD_OUT_SEEDS,
        HELD_OUT_QUIET,
        [*SEEDS, *QUIET_SEEDS],
        toy["tmp"] / "sweep",
        workers=1,
        n_boot=200,
    )


def test_only_a_confirmed_fix_enters_the_contract_report(
    toy: dict[str, Any], held: fix.HeldOut
) -> None:
    inputs, conf, winner = toy["inputs"], toy["conf"], toy["winner"].candidate
    report, base_detail = build_report(toy["result"], n_boot=50)
    detail = fix.fix_detail(inputs, winner, "toy_sync_fixed", conf, toy["rows"], 10, 2)
    assert detail.verdict.startswith("fix:")  # the in-sample wording, kept and labelled in-sample

    # in-sample only: stays in the sidecar
    r0, d0 = fix.add_fix_to_report(report, base_detail, inputs, winner, "toy_sync_fixed", detail)
    assert [c.config_name for c in r0.configs] == ["toy_sync"] and d0.fix is not None
    assert d0.fix.status == "in-sample only" and d0.fix.held_out is None

    for verdict in (fix.CANDIDATE, fix.CONFIRMED):
        forced = dataclasses.replace(held, verdict=verdict)
        r, d = fix.add_fix_to_report(
            report, base_detail, inputs, winner, "toy_sync_fixed", detail, forced
        )
        names = [c.config_name for c in r.configs]
        assert d.fix is not None and d.fix.status == verdict and d.fix.held_out is not None
        assert d.fix.held_out["n_seeds"] == len(HELD_OUT_SEEDS)
        if verdict == fix.CANDIDATE:
            assert (
                names == ["toy_sync"] and "toy_sync_fixed" not in d.configs
            )  # a candidate stays out
            continue
        assert names == ["toy_sync", "toy_sync_fixed"]
        fixed = r.config("toy_sync_fixed")
        assert fixed.cost_per_hour == r.config("toy_sync").cost_per_hour
        assert fixed.n_episodes == len(HELD_OUT_SEEDS) * len(held.tactics)  # the held-out run
        assert fixed.pd_at_operating_point == held.fixed.score.pd
        assert fixed.paired_vs_baseline == held.deltas
        assert [p.metric for p in fixed.paired_vs_baseline] == [
            METRIC_PD,
            METRIC_WORST,
            METRIC_DECISIONS,
        ]
        assert d.configs["toy_sync_fixed"].drones_down_s_per_hour == 0.0
    fleet_path, params_path = fix.write_fixed_fleet(winner, "toy_sync_fixed", toy["tmp"] / "fixed")
    written = FleetConfig.model_validate_json(fleet_path.read_text())
    assert written.name == "toy_sync_fixed" and written.agents == inputs.fleets["toy_sync"].agents
    assert json.loads(params_path.read_text())["env"] == {
        "AIRTIGHT_WEIGHT_MODE": "asset",
        "AIRTIGHT_ENGINE": "v0",
    }


def test_held_out_split_never_overlaps_the_search() -> None:
    seeds_file = Path(__file__).parents[2] / "data" / "seeds.json"
    everything = json.loads(seeds_file.read_text())["seeds"]
    in_sample, in_sample_quiet = everything[:50], everything[-20:]
    search, search_quiet = everything[:30], everything[-10:]
    intrusion, quiet = fix.held_out_split(seeds_file, 50, 20, 10)
    assert (len(intrusion), len(quiet)) == (130, 10)
    assert not set(intrusion) & (
        set(search) | set(search_quiet) | set(in_sample) | set(in_sample_quiet)
    )
    assert not set(quiet) & (set(search) | set(search_quiet)) and not set(quiet) & set(intrusion)
    assert set(quiet) <= set(in_sample_quiet)  # quiet nights the sweep ran and the search did not
    assert fix.held_out_split(seeds_file, 50, 20, 10, 40)[0] == intrusion[:40]
    with pytest.raises(ValueError, match="only 130 exist"):
        fix.held_out_split(seeds_file, 50, 20, 10, 150)
    with pytest.raises(ValueError, match="every quiet seed"):
        fix.held_out_split(seeds_file, 50, 20, 20)


def test_overlap_with_the_search_raises(toy: dict[str, Any]) -> None:
    fix.assert_disjoint([1, 2], [3], [4, 5])
    with pytest.raises(ValueError, match="overlap seeds the search saw"):
        fix.assert_disjoint([1, 2], [3], [2, 9])
    with pytest.raises(ValueError, match="overlap seeds the search saw"):
        fix.assert_disjoint([1, 2], [3], [3])  # a quiet seed the search used
    with pytest.raises(ValueError, match="both intrusion and quiet"):
        fix.assert_disjoint([1, 2], [2], [])
    with pytest.raises(ValueError, match="overlap seeds the search saw"):
        fix.held_out_confirmation(
            toy["inputs"],
            toy["winner"].candidate,
            toy["conf"],
            [SEEDS[0], 3000],
            HELD_OUT_QUIET,
            [*SEEDS, *QUIET_SEEDS],
            toy["tmp"] / "sweep",
            workers=1,
        )


def test_verdict_needs_the_interval_to_clear_zero() -> None:
    from airtight.contracts import PairedDelta

    def worst(lo: float, hi: float) -> PairedDelta:
        return PairedDelta(metric=METRIC_WORST, delta=(lo + hi) / 2, ci=(lo, hi))

    assert fix.verdict_for(worst(0.0, 0.29)) == fix.CANDIDATE  # touches 0, as F1's did in-sample
    assert fix.verdict_for(worst(-0.05, 0.30)) == fix.CANDIDATE
    assert fix.verdict_for(worst(0.01, 0.29)) == fix.CONFIRMED
    assert fix.verdict_for(worst(-0.30, -0.01)) == fix.CANDIDATE  # significantly WORSE is not a fix


def test_held_out_scores_the_tactics_chosen_in_sample(
    toy: dict[str, Any], held: fix.HeldOut
) -> None:
    conf = toy["conf"]
    assert held.in_sample_worst == (
        conf.fixed_score.worst_tactic_id,
        conf.baseline_score.worst_tactic_id,
    )
    assert held.seeds == HELD_OUT_SEEDS and held.quiet_seeds == HELD_OUT_QUIET
    assert [t.id for t in held.tactics] == [t.id for t in conf.tactics]  # the same tactic set
    fixed_id, base_id = held.in_sample_worst
    by = {d.metric: d for d in held.deltas}
    assert by[METRIC_WORST].delta == pytest.approx(
        held.fixed.score.pd_by_tactic[fixed_id] - held.baseline.score.pd_by_tactic[base_id]
    )
    own = {d.metric: d for d in held.deltas_held_out_worst}
    assert own[METRIC_WORST].delta == pytest.approx(
        held.fixed.score.worst_tactic_pd - held.baseline.score.worst_tactic_pd
    )
    assert by[METRIC_PD] == own[METRIC_PD]  # only the worst-tactic row depends on the choice
    assert held.verdict == fix.verdict_for(by[METRIC_WORST])
    detail = fix.held_out_detail(held)
    assert detail["worst_tactic_chosen_in_sample"]["baseline"]["tactic"] == base_id
    assert detail["seeds_never_seen_by_the_search"] is True


def test_ingredients_both_row_is_the_fix_exactly(toy: dict[str, Any], held: fix.HeldOut) -> None:
    rows = fix.ingredients(
        toy["inputs"], toy["winner"].candidate, held, toy["tmp"] / "sweep", workers=1, n_boot=200
    )
    assert [r[0].label for r in rows] == [
        "baseline",
        "asset weight only",
        "charge offsets only",
        "both (the fix)",
    ]
    baseline, _, _, both = rows
    assert baseline[0] is held.baseline and both[0] is held.fixed  # the same scored objects
    assert both[0].score == held.fixed.score
    assert both[1] == held.deltas_held_out_worst  # and the same paired deltas, exactly
    assert all(
        d.delta == 0.0 and d.ci == (0.0, 0.0) for d in baseline[1]
    )  # baseline against itself
    # the toy's winner keeps asset weight, so "weight only" IS the baseline and "offsets only" IS the fix
    assert rows[1][0].score.pd == held.baseline.score.pd
    assert rows[2][0].score.pd == held.fixed.score.pd
    table = fix.format_ingredients(rows)
    assert len(table.splitlines()) == 5 and "both (the fix)" in table
    assert [r["label"] for r in fix.ingredient_rows(rows)] == [r[0].label for r in rows]


def test_apply_policy_moves_only_the_go2_offset() -> None:
    fleet = _example_fleet()
    staggered = fleet.model_copy(
        update={
            "charge_policy": fleet.charge_policy.model_copy(
                update={"stagger_offsets_s": {"drone_2": 1950.0}}
            )
        }
    )
    winner = next(
        c for c in make_candidates(fleet) if c.go2_half_cycle and c.weight_mode == "uniform"
    )
    shifted = fix.apply_policy(staggered, winner)
    assert shifted.charge_policy.stagger_offsets_s == {"drone_2": 1950.0, "go2_1": 4500.0}
    assert shifted.agents == fleet.agents and shifted.cost_per_hour() == fleet.cost_per_hour()
    assert shifted.name == f"{fleet.name}_policy"
    no_shift = next(c for c in make_candidates(fleet) if not c.go2_half_cycle)
    assert fix.apply_policy(shifted, no_shift).charge_policy.stagger_offsets_s == {
        "drone_2": 1950.0
    }


def test_policy_by_fleet_is_paired_and_reuses_the_fix(toy: dict[str, Any]) -> None:
    inputs, winner = toy["inputs"], toy["winner"].candidate
    rows = fix.policy_by_fleet(
        inputs,
        winner,
        ["toy_sync"],
        SEEDS,
        QUIET_SEEDS,
        toy["tmp"] / "sweep",
        workers=1,
        n_boot=200,
        fixed_name="toy_sync_fixed",
    )
    ((name, naive, policy, deltas),) = rows
    assert name == "toy_sync" and policy.fleet.name == "toy_sync_fixed"
    assert naive.fleet.cost_per_hour() == policy.fleet.cost_per_hour()
    assert list(naive.score.pd_by_tactic) == list(policy.score.pd_by_tactic)  # the same tactics
    assert policy.score == toy["conf"].fixed_score and naive.score == toy["conf"].baseline_score
    assert [d.metric for d in deltas] == [METRIC_PD, METRIC_WORST, METRIC_DECISIONS]
    assert fix.policy_rows(rows)[0]["n_drones"] == 2 and "toy_sync" in fix.format_policy(rows)


def _ep(seed: int, peak: float) -> EpisodeScores:
    return EpisodeScores(seed, peak, None, {}, None, 0.02, 52.0, 27.0, 62.0, 3)


def test_find_pairs_only_returns_miss_then_timely() -> None:
    base = [
        _ep(1, NEVER_SEEN),
        _ep(2, 1.0),
        _ep(3, 5.0),
        _ep(4, 1.9),
        _ep(5, NEVER_SEEN),
        _ep(6, 0.5),
    ]
    fixed = [_ep(1, 9.0), _ep(2, 2.5), _ep(3, 9.0), _ep(4, 2.9), _ep(5, NEVER_SEEN), _ep(6, 3.0)]
    # thresholds 2.0 and 3.0: seed 1 (miss, then 9.0) and seed 6 (miss, then exactly 3.0) qualify.
    # seed 2: the fix scores 2.5, below ITS threshold. seed 3: the baseline already caught it.
    # seed 4: below the fix's threshold. seed 5: never seen by either.
    assert find_pairs(base, fixed, 2.0, 3.0) == [1, 6]  # strongest catch first
    assert find_pairs(base, fixed, 2.0, 100.0) == []
    assert find_pairs(base, base, 2.0, 2.0) == []  # a configuration against itself: never
    with pytest.raises(ValueError, match="aligned"):
        find_pairs(base, fixed[::-1], 2.0, 3.0)


def test_exported_pairs_really_are_miss_then_timely(
    toy: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRTIGHT_ENGINE", "stub")
    monkeypatch.delenv("AIRTIGHT_WEIGHT_MODE", raising=False)
    conf, inputs = toy["conf"], toy["inputs"]
    rows, note = export_pairs(inputs, toy["winner"].candidate, conf, toy["tmp"] / "pairs", n=3)
    import os

    assert os.environ["AIRTIGHT_ENGINE"] == "stub" and "AIRTIGHT_WEIGHT_MODE" not in os.environ
    assert 1 <= len(rows) <= 3 and conf.baseline_score.worst_tactic_id in note
    index = json.loads((toy["tmp"] / "pairs" / "index.json").read_text())
    assert index["pairs"] == rows and index["tactic"] == note
    tactic_id = rows[0]["tactic_id"]
    base = {e.seed: e for e in conf.baseline.episodes["toy_sync"][tactic_id]}
    fixed = {e.seed: e for e in conf.fixed.episodes["toy_sync_fixed"][tactic_id]}
    for row in rows:
        seed = row["seed"]
        assert base[seed].intruder_peak < conf.baseline_score.tau  # a miss under the baseline
        assert fixed[seed].intruder_peak >= conf.fixed_score.tau  # a catch under the fix
        assert row["baseline"]["detected_at_operating_threshold"] is False
        assert row["fixed"]["detected_at_operating_threshold"] is True
        for side, name in (("baseline", "toy_sync"), ("fixed", "toy_sync_fixed")):
            header, events = read_episode_log(Path(row[side]["log_path"]))
            kinds = [e.kind for e in events]
            assert (
                header.sim_version == "v0" and header.seed == seed and header.tactic.id == tactic_id
            )
            assert row[side]["configuration"] == name and kinds[-1] == "outcome"
    assert rows[0]["baseline"]["log_path"] != rows[0]["fixed"]["log_path"]


def test_weight_mode_env_name_matches_the_engine() -> None:
    assert fix.WEIGHT_MODE_ENV == episode.WEIGHT_MODE_ENV == "AIRTIGHT_WEIGHT_MODE"
    engine_modes = set(__import__("airtight.sim.geometry", fromlist=["x"]).WEIGHT_MODES)
    # "mix" is the campaign's continuous mixture; the fix loop searches the named modes only.
    assert set(fix.WEIGHT_MODES) == engine_modes - {"mix"}
