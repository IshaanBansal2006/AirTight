"""The policy plumbing: AIRTIGHT_PARAMS_JSON, the mixture weight, and dock assignment."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from airtight.sim import adapt
from airtight.sim.episode import (
    PARAMS_JSON_ENV,
    TASK_TIME_ENV,
    WEIGHT_MODE_ENV,
    EpisodeParams,
    official_params,
    params_from_mapping,
    params_to_mapping,
    simulate,
)
from airtight.sim.geometry import Grid, inside_mask, patrol_weight

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (PARAMS_JSON_ENV, TASK_TIME_ENV, WEIGHT_MODE_ENV):
        monkeypatch.delenv(name, raising=False)


def _weights(site: Site, **kwargs: object) -> np.ndarray:
    grid = Grid(*adapt.bounds(site), 5.0)
    inside = inside_mask(grid, adapt.perimeter(site))
    return patrol_weight(
        grid,
        inside,
        adapt.assets(site),
        r_c=adapt.critical_radius_m(site, 2.5),
        entry_positions=adapt.entries(site),
        **kwargs,  # type: ignore[arg-type]
    )


def test_named_modes_are_bit_identical_special_cases_of_the_mixture(yard_site: Site) -> None:
    asset = _weights(yard_site, mode="asset", asset_gain=0.7)
    assert np.array_equal(_weights(yard_site, mode="mix", asset_gain=0.7), asset)
    uniform = _weights(yard_site, mode="uniform")
    assert np.array_equal(_weights(yard_site, mode="mix", asset_gain=0.0), uniform)
    band = _weights(yard_site, mode="band", asset_gain=1.3)
    assert np.array_equal(_weights(yard_site, mode="mix", asset_gain=0.0, band_gain=1.3), band)


def test_named_modes_ignore_the_mixture_gains(yard_site: Site) -> None:
    for mode in ("asset", "uniform", "band"):
        plain = _weights(yard_site, mode=mode)
        assert np.array_equal(_weights(yard_site, mode=mode, entry_gain=5.0, band_gain=2.0), plain)


def test_entries_term_peaks_at_each_entry_and_is_zero_outside(yard_site: Site) -> None:
    grid = Grid(*adapt.bounds(yard_site), 5.0)
    inside = inside_mask(grid, adapt.perimeter(yard_site))
    base = _weights(yard_site, mode="mix", asset_gain=0.0)
    extra = _weights(yard_site, mode="mix", asset_gain=0.0, entry_gain=2.0) - base
    assert np.all(extra[~inside] == 0) and np.all(extra[inside] > 0)
    centres = grid.cell_centers()
    for ex, ey in adapt.entries(yard_site):
        distance = np.where(inside, np.hypot(centres[..., 0] - ex, centres[..., 1] - ey), np.inf)
        nearest = np.unravel_index(int(np.argmin(distance)), distance.shape)
        ring = (distance > 20.0) & (distance < 30.0)
        assert extra[nearest] > extra[ring].max()


def test_mixture_rejects_negative_gains_and_missing_entries(yard_site: Site) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _weights(yard_site, mode="mix", entry_gain=-0.1)
    grid = Grid(*adapt.bounds(yard_site), 5.0)
    inside = inside_mask(grid, adapt.perimeter(yard_site))
    with pytest.raises(ValueError, match="entry_positions"):
        patrol_weight(grid, inside, adapt.assets(yard_site), mode="mix", entry_gain=1.0)


def test_params_json_sets_fields_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = dataclasses.replace(
        EpisodeParams(battery=True),
        weight_mode="mix",
        entry_gain=1.5,
        band_gain=0.25,
        d0_m=40.0,
        dock_assignment=(("d0", "dock_a"),),
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(params_to_mapping(policy)))
    monkeypatch.setenv(PARAMS_JSON_ENV, str(path))
    assert official_params() == policy
    assert official_params().battery is True


def test_params_json_names_every_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"d0": 3, "battery": False, "task_time_s": 60, "top_fraction": "x"}))
    monkeypatch.setenv(PARAMS_JSON_ENV, str(path))
    with pytest.raises(ValueError) as err:
        official_params()
    text = str(err.value)
    assert "unknown field 'd0'" in text and "'battery' may not" in text
    assert "'task_time_s' may not" in text and "'top_fraction' must be a number" in text
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        official_params()
    monkeypatch.setenv(PARAMS_JSON_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(ValueError, match="cannot be read"):
        official_params()


def test_weight_mode_variable_is_an_alias_and_may_not_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WEIGHT_MODE_ENV, "uniform")
    alias = official_params()
    assert alias == params_from_mapping({"weight_mode": "uniform"}, EpisodeParams(battery=True))
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"weight_mode": "uniform", "d0_m": 30}))
    monkeypatch.setenv(PARAMS_JSON_ENV, str(path))
    assert official_params() == dataclasses.replace(alias, d0_m=30.0)
    path.write_text(json.dumps({"weight_mode": "band"}))
    with pytest.raises(ValueError, match="disagrees"):
        official_params()


def test_task_time_variable_is_separate_and_marks_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from airtight.sim.episode import assumption_task_time_s

    assert assumption_task_time_s() == 0.0 and official_params().task_time_s == 0.0
    monkeypatch.setenv(TASK_TIME_ENV, "60")
    assert assumption_task_time_s() == 60.0
    assert official_params() == EpisodeParams(battery=True, task_time_s=60.0)
    for bad in ("soon", "-1", "inf"):
        monkeypatch.setenv(TASK_TIME_ENV, bad)
        with pytest.raises(ValueError, match=TASK_TIME_ENV):
            official_params()


def test_dock_assignment_moves_only_the_listed_agent(
    yard_site: Site, fleet_of: Callable[..., FleetConfig]
) -> None:
    fleet = fleet_of(3)
    docks = adapt.dock_ids(yard_site)
    if len(docks) < 2:
        pytest.skip("the scenario needs two docks for this test")
    default = adapt.default_dock_assignment(yard_site, fleet)
    assert default == {f"d{i}": docks[i % len(docks)] for i in range(3)}
    same = {a: adapt.start_position(yard_site, fleet, a, default) for a in default}
    for agent_id, pos in same.items():
        assert np.array_equal(pos, adapt.start_position(yard_site, fleet, agent_id))
    other = next(d for d in docks if d != default["d0"])
    moved = adapt.start_position(yard_site, fleet, "d0", {"d0": other})
    dock_xy = adapt.docks(yard_site)[docks.index(other)]
    assert np.isclose(np.linalg.norm(moved - dock_xy), adapt.START_OFFSET_M)
    assert np.array_equal(
        adapt.start_position(yard_site, fleet, "d1", {"d0": other}),
        adapt.start_position(yard_site, fleet, "d1"),
    )
    with pytest.raises(ValueError, match="not on the site"):
        adapt.start_position(yard_site, fleet, "d0", {"d0": "moon_base"})
    with pytest.raises(ValueError, match="not in fleet"):
        adapt.start_position(yard_site, fleet, "d0", {"stranger": docks[0]})


def test_default_dock_assignment_and_named_mixture_give_identical_episodes(
    yard_site: Site,
    yard_curve: SensorCurves,
    yard_tactic: Tactic,
    fleet_of: Callable[..., FleetConfig],
) -> None:
    fleet = fleet_of(2)
    plain = simulate(yard_site, fleet, yard_tactic, yard_curve, 7, EpisodeParams(battery=True))
    explicit = dataclasses.replace(
        EpisodeParams(battery=True),
        weight_mode="mix",
        dock_assignment=tuple(sorted(adapt.default_dock_assignment(yard_site, fleet).items())),
    )
    assert simulate(yard_site, fleet, yard_tactic, yard_curve, 7, explicit) == plain
