"""The campaign's pure pieces: sizing, tactic sets, targets, upgrades, the frozen-copy guard."""

from __future__ import annotations

import pytest

from airtight.score import campaign, hardware
from airtight.score.adversary import Limits
from airtight.sim import scenarios

LIMITS = Limits(0.8, 2.0, 6, 1.5, 1.0, 2.0)


def test_shares_leave_room_and_stages_follow_priority() -> None:
    assert sum(campaign.SHARES.values()) <= 1.0
    assert campaign.STAGES[0] == "probe" and campaign.STAGES[1] == "A"
    order = list(campaign.STAGES)
    assert order.index("search") < order.index("validation") < order.index("final_standin")
    assert order.index("final_strong") < order.index("sensitivity") < order.index("audit")


def test_fit_respects_bounds_and_fixed_costs() -> None:
    assert campaign._fit(1000.0, 10.0, 0.0, (5, 50)) == 50
    assert campaign._fit(1000.0, 10.0, 800.0, (5, 50)) == 20
    assert campaign._fit(10.0, 10.0, 800.0, (5, 50)) == 5
    assert campaign._fit(10.0, 0.0, 0.0, (5, 50)) == 50


def test_tactic_sets_have_both_speed_limits_and_unique_ids() -> None:
    site = scenarios.load_site()
    n_entries = len(site.entry_points)
    reduced = campaign.reduced_common(site, LIMITS)
    standin = campaign.standin_common(site, LIMITS)
    assert len(reduced) == n_entries * 6 and len(standin) == n_entries * 24
    for tactics in (reduced, standin):
        assert len({t.id for t in tactics}) == len(tactics)
        assert len({t.content_hash() for t in tactics}) == len(tactics)
        assert {t.speed_mps for t in tactics} == {LIMITS.speed_min_mps, LIMITS.speed_cap_mps}
        assert all(0.0 <= t.phase < 1.0 and t.waypoints == [site.asset] for t in tactics)


def test_with_gaps_attacks_every_gap_once() -> None:
    site = scenarios.load_site()
    fleet = scenarios.load_fleet("2drones")
    common = campaign.standin_common(site, LIMITS)
    once = campaign.with_gaps(common, site, [fleet], LIMITS.speed_cap_mps)
    gaps = [t for t in once if t.id.startswith("gap-")]
    assert gaps, "a fleet whose drones charge together has a gap"
    assert len({(t.entry_id, t.phase) for t in gaps}) == len(gaps)
    again = campaign.with_gaps(once, site, [fleet], LIMITS.speed_cap_mps)
    assert [t.id for t in again] == [t.id for t in once]


def _row(label: str, cost: float, pd: float, worst: float) -> dict[str, object]:
    return {
        "label": label,
        "cost_per_hour": cost,
        "n_seeds": 40,
        "pd": pd,
        "pd_ci": [pd - 0.005, pd + 0.005],
        "worst_naive": {"pd": worst, "ci": [worst - 0.1, worst + 0.1], "tactic": f"t-{label}"},
    }


def test_cheapest_to_target_reports_the_cheapest_or_the_ceiling() -> None:
    rows = [_row("a", 50.0, 0.96, 0.5), _row("b", 40.0, 0.952, 0.7), _row("c", 30.0, 0.9, 0.2)]
    hit = campaign.cheapest_to_target(rows, "pd")
    assert hit["reached"] is True and hit["label"] == "b" and hit["cost_per_hour"] == 40.0
    assert hit["value"] == 0.952 and hit["n_seeds"] == 40
    # b reaches the target only on its point estimate; a is the cheapest whose interval does
    assert hit["confirmed_label"] == "a" and hit["confirmed_cost_per_hour"] == 50.0
    miss = campaign.cheapest_to_target(rows, "worst_naive")
    assert miss["reached"] is False and miss["ceiling"] == 0.7 and miss["label"] == "b"
    assert miss["worst_tactic_of_ceiling"] == "t-b"
    assert campaign.cheapest_to_target(rows, "worst_heldout") == {"reached": False, "ceiling": None}


def test_targets_never_name_an_ingredient_row() -> None:
    rows = [_row("baseline", 55.0, 0.5, 0.1), _row("ingredient:weight", 55.0, 0.99, 0.99)]
    targets = campaign._targets({"final_standin": {"rows": rows}})
    assert targets["strict"]["overall"]["reached"] is False
    assert targets["strict"]["overall"]["label"] == "baseline"


def test_duty_shares_compare_the_attack_window_with_steady_state() -> None:
    fleet = scenarios.load_fleet("2drones")
    shares = campaign.duty_shares(fleet)
    assert set(shares) == {a.id for a in fleet.agents if a.charge_time_s > 0}
    for share in shares.values():
        assert 0.0 <= share["in_window"] <= 1.0
        assert share["in_window"] == pytest.approx(share["steady_state"])


def test_upgrades_are_exactly_one_purchase_away() -> None:
    spec = hardware.HardwareSpec(n_drones=6, swap_docks=False, entry_cameras=True)
    assert campaign._upgrades(spec) == [hardware.HardwareSpec(6, True, True)]
    small = hardware.HardwareSpec(n_drones=1, swap_docks=False, entry_cameras=False)
    assert {u.name for u in campaign._upgrades(small)} == {
        hardware.HardwareSpec(2, False, False).name,
        hardware.HardwareSpec(1, True, False).name,
        hardware.HardwareSpec(1, False, True).name,
    }


def test_ingredient_groups_cover_every_policy_field_once() -> None:
    from airtight.score.policy import FLOAT_FIELDS

    grouped = [f for fields in campaign.INGREDIENTS.values() for f in fields]
    assert sorted(grouped) == sorted([*FLOAT_FIELDS, "offsets", "docks"])


def test_refuses_a_relative_out_and_code_outside_the_frozen_copy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit):
        campaign.main(["--out", "relative/path"])
    with pytest.raises(SystemExit, match="outside the frozen copy"):
        campaign.main(["--out", str(tmp_path / "campaign"), "--frozen-root", str(tmp_path)])


def test_refuses_policy_variables_in_the_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    import json

    from airtight.sim.episode import PARAMS_JSON_ENV

    scenario = tmp_path / "scenario"
    (scenario / "fleets").mkdir(parents=True)
    (scenario / "site.json").write_text(scenarios.load_site().model_dump_json())
    (scenario / "sensor_curve.json").write_text(scenarios.load_sensor_curves().model_dump_json())
    (scenario / "fleets" / "sweep.json").write_text(json.dumps({"baseline": "b", "configs": ["b"]}))
    (scenario / "fleets" / "b.json").write_text(scenarios.load_fleet("2drones").model_dump_json())
    (scenario / "redteam_config.json").write_text(
        json.dumps(
            {
                "speed_min_mps": 0.8,
                "speed_cap_mps": 2.0,
                "max_waypoints": 6,
                "perimeter_tol_m": 1.5,
                "asset_tol_m": 1.0,
                "min_leg_m": 2.0,
            }
        )
    )
    monkeypatch.setenv(PARAMS_JSON_ENV, str(tmp_path / "policy.json"))
    with pytest.raises(SystemExit, match=PARAMS_JSON_ENV):
        campaign.main(
            ["--out", str(tmp_path / "campaign"), "--scenario-dir", str(scenario), "--smoke"]
        )
