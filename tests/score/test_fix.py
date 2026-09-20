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
    from airtight.score.sweep import SweepInputs, SweepResult

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
        own = own_gap_tactics(inputs, row.candidate.fleet, set())
        gaps = fleet_gaps(row.candidate.fleet)["drones_down"]
        assert len(own) == len(gaps) * len(inputs.site.entry_points)
        assert all(any(a <= t.phase < b for a, b in gaps) for t in own)
        assert all(t.speed_mps == 1.4 and t.id.startswith("own-") for t in own)
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
    result: SweepResult = toy["result"]
    report, base_detail = build_report(result, n_boot=50)
    new_report, new_detail = fix.add_fix_to_report(
        report, base_detail, inputs, same, "toy_sync_same", conf, detail
    )
    assert [c.config_name for c in new_report.configs] == [
        "toy_sync"
    ]  # nothing added to the contract
    assert (
        new_detail.fix is not None
        and new_detail.fix.is_fix is False
        and len(new_detail.fix.search) == 3
    )


def test_a_real_fix_enters_the_report_with_paired_deltas(toy: dict[str, Any]) -> None:
    inputs, conf, winner = toy["inputs"], toy["conf"], toy["winner"].candidate
    report, base_detail = build_report(toy["result"], n_boot=50)
    detail = fix.fix_detail(inputs, winner, "toy_sync_fixed", conf, toy["rows"], 10, 2)
    new_report, new_detail = fix.add_fix_to_report(
        report, base_detail, inputs, winner, "toy_sync_fixed", conf, detail
    )
    fixed = new_report.config("toy_sync_fixed")
    assert fixed.cost_per_hour == new_report.config("toy_sync").cost_per_hour
    assert [p.metric for p in fixed.paired_vs_baseline] == [
        METRIC_PD,
        METRIC_WORST,
        METRIC_DECISIONS,
    ]
    assert fixed.coverage_gap_s_per_hour < new_report.config("toy_sync").coverage_gap_s_per_hour
    assert new_detail.configs["toy_sync_fixed"].drones_down_s_per_hour == 0.0
    assert new_detail.fix is not None and new_detail.fix.verdict.startswith("fix:")
    assert new_detail.fix.what_changed["stagger_offsets_s"] == {"d1": 1800.0}
    fleet_path, params_path = fix.write_fixed_fleet(winner, "toy_sync_fixed", toy["tmp"] / "fixed")
    written = FleetConfig.model_validate_json(fleet_path.read_text())
    assert written.name == "toy_sync_fixed" and written.agents == inputs.fleets["toy_sync"].agents
    assert json.loads(params_path.read_text())["env"] == {
        "AIRTIGHT_WEIGHT_MODE": "asset",
        "AIRTIGHT_ENGINE": "v0",
    }


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
    assert set(fix.WEIGHT_MODES) == set(
        __import__("airtight.sim.geometry", fromlist=["x"]).WEIGHT_MODES
    )
