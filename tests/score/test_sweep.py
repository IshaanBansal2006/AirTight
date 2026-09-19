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


def test_stand_in_grid_and_the_check_on_its_claim(scenario_dir: Path) -> None:
    inputs = load_inputs(scenario_dir)  # no tactic directories: the stand-in adversary
    site = inputs.site
    assert inputs.tactic_source.startswith(sweep.GRID_SOURCE)
    assert len(inputs.tactics) == len(site.entry_points) * sweep.GRID_PHASES
    first = inputs.tactics[0]
    assert first.id == f"grid-{site.entry_points[0].id}-0.0000" and first.speed_mps == 2.5
    assert first.waypoints == [site.asset] and len({t.id for t in inputs.tactics}) == len(
        inputs.tactics
    )
    assert grid_tactics(site, 2.5) == inputs.tactics

    hits = grid_hits(inputs)
    # synchronized: nobody up for phases 0.417 to 1.0, which holds grid phases 7/16 .. 15/16
    assert (
        hits["2drones"]["uncovered_phases_hit"] == 9
        and hits["2drones"]["uncovered_gaps_missed"] == 0
    )
    # staggered: two 300 s gaps, 0.417-0.5 and 0.917-1.0; the grid lands in each (7/16 and 15/16)
    assert hits["2drones_staggered"]["uncovered_phases_hit"] == 2
    assert hits["2drones_staggered"]["uncovered_gaps_missed"] == 0
    assert "WARNING" not in sweep.format_grid_hits(inputs)

    coarse = dataclasses.replace(inputs, tactics=grid_tactics(site, 2.5, n_phases=4))
    assert grid_hits(coarse)["2drones_staggered"]["uncovered_gaps_missed"] == 2
    assert "WARNING" in sweep.format_grid_hits(coarse)


def test_no_tactics_and_no_speed_cap_is_an_error(scenario_dir: Path) -> None:
    (scenario_dir / "redteam_config.json").unlink()
    with pytest.raises(ValueError, match="speed cap"):
        load_inputs(scenario_dir)
    assert load_inputs(scenario_dir, speed_cap_mps=2.0).tactics[0].speed_mps == 2.0
