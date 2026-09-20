from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from airtight.score import sweep
from airtight.score.sweep import engine_tag, grid_hits, grid_tactics, load_inputs, run_sweep
from airtight.sim import scenarios
from airtight.sim.episode import EpisodeParams, official_params, simulate, simulate_quiet

if TYPE_CHECKING:
    from pathlib import Path

SEEDS = [1000, 1001, 1002, 1003, 1004]
QUIET_SEEDS = [2000, 2001]


def _dump(model, path: Path) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2))


@pytest.fixture
def scenario_dir(tmp_path: Path) -> Path:
    """yard_night laid out the way lane C lays out a scenario."""
    root = tmp_path / "scenario"
    _dump(scenarios.load_site(), root / "site.json")
    _dump(scenarios.load_sensor_curves(), root / "sensor_curve.json")
    for name in ("2drones", "2drones_staggered"):
        _dump(scenarios.load_fleet(name), root / "fleets" / f"{name}.json")
    spec = {"baseline": "2drones", "configs": ["2drones", "2drones_staggered"]}
    (root / "fleets" / "sweep.json").write_text(json.dumps(spec))
    (root / "redteam_config.json").write_text(json.dumps({"speed_cap_mps": 2.5}))
    return root


@pytest.fixture
def tactics_dir(tmp_path: Path) -> Path:
    root = tmp_path / "tactics"
    _dump(
        scenarios.load_tactic("jog").model_copy(update={"phase": 0.2, "id": "jog-up"}),
        root / "a.json",
    )
    pair = [scenarios.load_tactic("walk").model_copy(update={"phase": 0.7, "id": "walk-down"})]
    (root / "b_list.json").write_text(json.dumps([json.loads(t.model_dump_json()) for t in pair]))
    return root


def _files(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*.jsonl"))}


def test_loads_lane_cs_layout_in_spec_order(scenario_dir: Path, tactics_dir: Path) -> None:
    inputs = load_inputs(scenario_dir, [tactics_dir])
    assert inputs.baseline == "2drones" and list(inputs.fleets) == ["2drones", "2drones_staggered"]
    assert [t.id for t in inputs.tactics] == ["jog-up", "walk-down"]  # a file may hold a list
    assert inputs.speed_cap_mps == 2.5 and str(tactics_dir) in inputs.tactic_source


def test_sweep_results_cache_layout_and_values(
    scenario_dir: Path, tactics_dir: Path, tmp_path: Path
) -> None:
    inputs = load_inputs(scenario_dir, [tactics_dir])
    out = tmp_path / "sweep"
    result = run_sweep(inputs, SEEDS, QUIET_SEEDS, out, workers=1)
    assert result.n_computed == 2 * 2 * 5 + 2 * 2
    tag = engine_tag(official_params(), inputs.curves)
    fleet = inputs.fleets["2drones"]
    base = out / tag / inputs.site.content_hash() / fleet.content_hash()
    assert sorted(p.name for p in base.glob("*.jsonl")) == ["jog-up.jsonl", "walk-down.jsonl"]
    assert (base / "quiet" / "quiet.jsonl").is_file()
    assert not list(out.rglob("*.tmp"))  # atomic writes leave nothing behind

    jog = inputs.tactics[0]
    direct = [simulate(inputs.site, fleet, jog, inputs.curves, s, official_params()) for s in SEEDS]
    assert result.episodes["2drones"]["jog-up"] == direct  # simulate with official_params, exactly
    quiet = [simulate_quiet(inputs.site, fleet, inputs.curves, s) for s in QUIET_SEEDS]
    assert result.quiet["2drones"] == quiet
    lines = (base / "jog-up.jsonl").read_text().splitlines()
    assert [json.loads(line)["seed"] for line in lines] == SEEDS
    assert {k for k in json.loads(lines[0]) if k != "_guard"} == {
        f.name for f in dataclasses.fields(direct[0])
    }


def test_interrupt_and_resume_is_byte_identical_and_a_rerun_computes_nothing(
    scenario_dir: Path, tactics_dir: Path, tmp_path: Path
) -> None:
    inputs = load_inputs(scenario_dir, [tactics_dir])
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    run_sweep(inputs, SEEDS, QUIET_SEEDS, whole, workers=1)

    first = run_sweep(inputs, SEEDS[:2], QUIET_SEEDS[:1], resumed, workers=1)  # "interrupted"
    assert first.n_computed == 2 * 2 * 2 + 2
    second = run_sweep(inputs, SEEDS, QUIET_SEEDS, resumed, workers=1)
    assert second.n_computed == 2 * 2 * 3 + 2  # only what was missing
    assert _files(resumed) == _files(whole)

    again = run_sweep(inputs, SEEDS, QUIET_SEEDS, resumed, workers=1)
    assert again.n_computed == 0 and again.episodes == second.episodes
    assert _files(resumed) == _files(whole)


def test_pool_equals_serial(scenario_dir: Path, tactics_dir: Path, tmp_path: Path) -> None:
    inputs = load_inputs(scenario_dir, [tactics_dir])
    serial = run_sweep(inputs, SEEDS, QUIET_SEEDS, tmp_path / "serial", workers=1)
    pooled = run_sweep(inputs, SEEDS, QUIET_SEEDS, tmp_path / "pooled", workers=2)
    assert pooled.episodes == serial.episodes and pooled.quiet == serial.quiet
    assert _files(tmp_path / "pooled") == _files(tmp_path / "serial")


def test_a_changed_tactic_under_the_same_id_is_not_served_from_cache(
    scenario_dir: Path, tactics_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sweep"
    before = run_sweep(load_inputs(scenario_dir, [tactics_dir]), SEEDS, QUIET_SEEDS, out, workers=1)
    edited = scenarios.load_tactic("jog").model_copy(update={"phase": 0.6, "id": "jog-up"})
    _dump(edited, tactics_dir / "a.json")
    after = run_sweep(load_inputs(scenario_dir, [tactics_dir]), SEEDS, QUIET_SEEDS, out, workers=1)
    assert after.n_computed == 2 * 5  # the edited tactic, for both configurations, and nothing else
    assert after.episodes["2drones"]["jog-up"] != before.episodes["2drones"]["jog-up"]
    assert after.episodes["2drones"]["walk-down"] == before.episodes["2drones"]["walk-down"]


def test_every_bad_file_is_named_in_one_error(scenario_dir: Path, tactics_dir: Path) -> None:
    (tactics_dir / "broken.json").write_text('{"id": "x", "family": "teleport"}')
    (tactics_dir / "garbage.json").write_text("{not json")
    stray = scenarios.load_tactic("jog").model_copy(update={"id": "lost", "entry_id": "moon_gate"})
    _dump(stray, tactics_dir / "stray.json")
    (scenario_dir / "fleets" / "2drones_staggered.json").write_text('{"name": "nope"}')
    with pytest.raises(ValueError) as err:
        load_inputs(scenario_dir, [tactics_dir])
    message = str(err.value)
    for name in ("broken.json", "garbage.json", "stray.json", "2drones_staggered.json"):
        assert name in message
    assert "moon_gate" in message and "not a valid Tactic" in message


def test_engine_tag_changes_with_params_and_curve() -> None:
    curves = scenarios.load_sensor_curves()
    tag = engine_tag(official_params(), curves)
    assert tag == engine_tag(official_params(), curves) and tag.startswith("v")
    assert engine_tag(EpisodeParams(battery=True, warmup_s=151.0), curves) != tag
    assert engine_tag(EpisodeParams(battery=False), curves) != tag
    other = curves.model_copy(update={"source": "a different calibration"})
    assert engine_tag(official_params(), other) != tag


def test_stand_in_adversary_is_the_grid_plus_every_gap_midpoint(scenario_dir: Path) -> None:
    inputs = load_inputs(scenario_dir)  # no tactic directories: the stand-in adversary
    site = inputs.site
    assert inputs.tactic_source.startswith(sweep.GRID_SOURCE)
    grid = grid_tactics(site, 2.5)
    assert len(grid) == len(site.entry_points) * sweep.GRID_PHASES
    assert inputs.tactics[: len(grid)] == grid
    assert grid[0].id == f"grid-{site.entry_points[0].id}-0.0000" and grid[0].waypoints == [
        site.asset
    ]

    # yard_night gaps: synchronized 0.417-1.0 (midpoint 0.708); staggered 0.417-0.5 and 0.917-1.0
    # (midpoints 0.458 and 0.958). None is within 0.01 of a grid phase, so all three are added,
    # once each, for every entry, and every configuration faces the same full set.
    extra = inputs.tactics[len(grid) :]
    assert sorted({round(t.phase, 4) for t in extra}) == [0.4583, 0.7083, 0.9583]
    assert len(extra) == 3 * len(site.entry_points)
    assert {t.id for t in extra} == {
        f"gap-{e.id}-{ph:.4f}" for e in site.entry_points for ph in {t.phase for t in extra}
    }
    assert all(t.speed_mps == 2.5 and t.waypoints == [site.asset] for t in extra)
    assert len({t.id for t in inputs.tactics}) == len(inputs.tactics)

    hits = grid_hits(inputs)
    assert hits["2drones"]["uncovered_gaps_missed"] == 0
    assert hits["2drones_staggered"]["uncovered_gaps_missed"] == 0
    assert (
        hits["2drones_staggered"]["uncovered_phases_hit"] == 2 + 2
    )  # 7/16, 15/16 and two midpoints
    assert sweep.unattacked_gaps(inputs.fleets, inputs.tactics) == []
    assert "WARNING" not in sweep.format_grid_hits(inputs)


def _fleet(endurance: float, charge: float, offsets: dict[str, float], n: int = 2):  # type: ignore[no-untyped-def]
    base = scenarios.load_fleet("2drones")
    drone = base.agents[0].model_copy(update={"endurance_s": endurance, "charge_time_s": charge})
    agents = [drone.model_copy(update={"id": f"d{i}"}) for i in range(n)]
    policy = base.charge_policy.model_copy(update={"stagger_offsets_s": offsets})
    return base.model_copy(update={"agents": agents, "charge_policy": policy})


def test_a_grid_alone_steps_over_a_narrow_gap_and_the_check_says_so() -> None:
    # On duty 3500 s of a 3600 s cycle: one gap, phases 0.972 to 1.0, narrower than 1/16.
    narrow = _fleet(3500.0, 100.0, {})
    site = scenarios.load_site()
    grid = grid_tactics(site, 2.5)
    missed = sweep.unattacked_gaps({"narrow": narrow}, grid)
    assert len(missed) == 2 and all("narrow" in m and "0.972-1.000" in m for m in missed)
    assert {m.split(": ")[1].split(" gap")[0] for m in missed} == {"uncovered", "drones_down"}

    full = sweep.stand_in_tactics(site, [narrow], 2.5)
    assert sweep.unattacked_gaps({"narrow": narrow}, full) == []
    (mid,) = sweep.gap_phases([narrow], sorted({t.phase for t in grid}))
    assert mid == pytest.approx((3500.0 / 3600.0 + 1.0) / 2.0)


def test_midpoints_are_shared_across_configurations_and_deduplicated() -> None:
    grid_phases = [k / 16 for k in range(16)]
    sync, staggered = _fleet(1500.0, 2100.0, {}), _fleet(1500.0, 2100.0, {"d1": 1800.0})
    both = sweep.gap_phases([sync, staggered], grid_phases)
    assert both == pytest.approx([0.4583, 0.7083, 0.9583], abs=1e-4)
    assert sweep.gap_phases([sync, staggered, sync, staggered], grid_phases) == both  # no repeats
    # a midpoint that lands within 0.01 of a phase already in the set adds nothing
    assert sweep.gap_phases([sync], [*grid_phases, 0.7083 + 0.005]) == []


def test_a_sliver_gap_next_to_a_grid_phase_is_still_attacked() -> None:
    # The gap is phases 0.9385 to 0.9500. Its midpoint, 0.9443, is within 0.01 of the grid phase
    # 0.9375, which lies just OUTSIDE the gap. Deduplication alone would leave the gap empty.
    endurance = 0.9385 * 3600.0
    sliver = _fleet(endurance, 3600.0 - endurance, {"d1": 180.0})
    gaps = sweep.fleet_gaps(sliver)["uncovered"]
    assert gaps == [pytest.approx((0.9385, 0.95))]
    grid_phases = [k / 16 for k in range(16)]
    assert not any(gaps[0][0] <= p < gaps[0][1] for p in grid_phases)
    (added,) = sweep.gap_phases([sliver], grid_phases)
    assert gaps[0][0] <= added < gaps[0][1] and abs(added - 0.9375) < sweep.PHASE_DEDUP
    site = scenarios.load_site()
    assert (
        sweep.unattacked_gaps({"sliver": sliver}, sweep.stand_in_tactics(site, [sliver], 2.5)) == []
    )


def test_no_tactics_and_no_speed_cap_is_an_error(scenario_dir: Path) -> None:
    (scenario_dir / "redteam_config.json").unlink()
    with pytest.raises(ValueError, match="speed cap"):
        load_inputs(scenario_dir)
    assert load_inputs(scenario_dir, speed_cap_mps=2.0).tactics[0].speed_mps == 2.0
