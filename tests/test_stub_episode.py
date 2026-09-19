from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic, read_episode_log
from airtight.sim.runner import run_episode

EXAMPLES = resources.files("airtight.contracts.examples")


def _load(name: str, model: type):  # type: ignore[no-untyped-def]
    return model.model_validate_json(EXAMPLES.joinpath(name).read_text())


def test_stub_episode_has_contract_shape(tmp_path: Path) -> None:
    site, fleet = _load("site.json", Site), _load("fleet_config.json", FleetConfig)
    tactic, curves = _load("tactic.json", Tactic), _load("sensor_curve.json", SensorCurves)
    result = run_episode(site, fleet, tactic, curves, seed=7, log_dir=tmp_path)
    header, events = read_episode_log(result.log_path)
    outcome = list(events)[-1]
    assert header.seed == 7 and header.site_hash == site.content_hash()
    assert outcome.kind == "outcome" and outcome.timely_detected == result.timely_detected


def test_stub_episode_is_seed_deterministic(tmp_path: Path) -> None:
    site, fleet = _load("site.json", Site), _load("fleet_config.json", FleetConfig)
    tactic, curves = _load("tactic.json", Tactic), _load("sensor_curve.json", SensorCurves)
    a = run_episode(site, fleet, tactic, curves, seed=3, log_dir=tmp_path / "a")
    b = run_episode(site, fleet, tactic, curves, seed=3, log_dir=tmp_path / "b")
    assert (a.timely_detected, a.t_alarm, a.t_cdp) == (b.timely_detected, b.t_alarm, b.t_cdp)


def test_seed_list_committed() -> None:
    seeds = json.loads((Path(__file__).parents[1] / "data" / "seeds.json").read_text())["seeds"]
    assert len(seeds) == 200 and len(set(seeds)) == 200
