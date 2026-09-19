from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from airtight.contracts import (
    FleetConfig,
    Report,
    SensorCurves,
    Site,
    Tactic,
    read_episode_log,
)

EXAMPLES = resources.files("airtight.contracts.examples")


def _text(name: str) -> str:
    return EXAMPLES.joinpath(name).read_text()


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("site.json", Site),
        ("fleet_config.json", FleetConfig),
        ("tactic.json", Tactic),
        ("tactic_decoy.json", Tactic),
        ("tactic_blind_spot.json", Tactic),
        ("sensor_curve.json", SensorCurves),
        ("report.json", Report),
    ],
)
def test_example_loads(filename: str, model: type) -> None:
    instance = model.model_validate_json(_text(filename))
    assert model.model_validate_json(instance.model_dump_json()) == instance


def test_example_episode_log_round_trips(tmp_path: Path) -> None:
    src = Path(str(EXAMPLES.joinpath("episode.jsonl")))
    header, events = read_episode_log(src)
    kinds = [e.kind for e in events]
    assert header.tactic.family == "charging_window"
    assert kinds[0] == "battery" and kinds[-1] == "outcome"


def test_examples_cross_reference() -> None:
    site = Site.model_validate_json(_text("site.json"))
    fleet = FleetConfig.model_validate_json(_text("fleet_config.json"))
    curves = SensorCurves.model_validate_json(_text("sensor_curve.json"))
    for name in ("tactic.json", "tactic_decoy.json", "tactic_blind_spot.json"):
        assert site.entry(Tactic.model_validate_json(_text(name)).entry_id)
    assert {a.sensor_type for a in fleet.agents} <= set(curves.curves)
    assert {s.sensor_type for s in site.fixed_sensors} <= set(curves.curves)
    assert {r.cls for r in site.benign_routes} <= set.intersection(
        *(set(c.pfa_per_look_by_class) for c in curves.curves.values())
    )


def test_content_hash_is_stable() -> None:
    site = Site.model_validate_json(_text("site.json"))
    assert site.content_hash() == Site.model_validate_json(site.model_dump_json()).content_hash()
    assert len(site.content_hash()) == 12
