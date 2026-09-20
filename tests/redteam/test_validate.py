from __future__ import annotations

from airtight.contracts import XY, CommsEvent, Decoy, Site, Tactic
from airtight.redteam import RedTeamConfig, validate


def test_hand_examples_are_valid(site: Site, hand_tactics: list[Tactic]) -> None:
    for t in hand_tactics:
        assert validate(t, site) == [], t.id


def test_unknown_entry(site: Site, hand_tactics: list[Tactic]) -> None:
    t = hand_tactics[0].model_copy(update={"entry_id": "roof"})
    assert "roof" in validate(t, site)[0]


def test_speed_cap(site: Site, hand_tactics: list[Tactic]) -> None:
    t = hand_tactics[0].model_copy(update={"speed_mps": 9.0})
    assert any("speed" in e for e in validate(t, site))


def test_last_waypoint_must_be_asset(site: Site, hand_tactics: list[Tactic]) -> None:
    t = hand_tactics[0].model_copy(update={"waypoints": [XY(x=60, y=62), XY(x=58, y=50)]})
    assert any("asset" in e for e in validate(t, site))


def test_path_may_not_leave_perimeter(site: Site, hand_tactics: list[Tactic]) -> None:
    t = hand_tactics[0].model_copy(update={"waypoints": [XY(x=2, y=2), site.asset]})
    errors = validate(t, site)
    assert any("perimeter" in e or "bounds" in e for e in errors)


def test_family_specific_requirements(site: Site, hand_tactics: list[Tactic]) -> None:
    decoyless = hand_tactics[1].model_copy(update={"decoy": None})
    assert any("requires a decoy" in e for e in validate(decoyless, site))
    late_cut = hand_tactics[0].model_copy(
        update={"family": "comms_cut", "comms_event": CommsEvent(t_s=10_000, kind="cut_base_link")}
    )
    assert any("after the intruder reaches" in e for e in validate(late_cut, site))
    bad_lead = hand_tactics[1].model_copy(
        update={"decoy": Decoy(position=XY(x=60, y=68), lead_time_s=5)}
    )
    assert any("decoy lead" in e for e in validate(bad_lead, site, RedTeamConfig()))
