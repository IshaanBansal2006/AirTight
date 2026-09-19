from __future__ import annotations

import numpy as np
import pytest

from airtight.contracts import FleetConfig, SensorCurves, Site
from airtight.redteam import perturb, sample_tactic, validate
from airtight.redteam.coverage import GeometryCoverage
from airtight.redteam.families import FAMILIES, charge_cycle_s


@pytest.mark.parametrize("family", FAMILIES)
def test_sampled_tactics_are_valid_and_deterministic(
    family: str, site: Site, fleet: FleetConfig, curves: SensorCurves
) -> None:
    a = [sample_tactic(family, site, fleet, curves, np.random.default_rng(3)) for _ in range(5)]
    b = [sample_tactic(family, site, fleet, curves, np.random.default_rng(3)) for _ in range(5)]
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]
    for t in a:
        assert t.family == family and t.origin == "random"
        assert validate(t, site) == []


def test_families_carry_their_payload(site: Site, fleet: FleetConfig, curves: SensorCurves) -> None:
    rng = np.random.default_rng(1)
    assert sample_tactic("decoy", site, fleet, curves, rng).decoy is not None
    assert sample_tactic("comms_cut", site, fleet, curves, rng).comms_event is not None
    assert sample_tactic("charging_window", site, fleet, curves, rng).decoy is None


def test_blind_spot_paths_use_unwatched_cells(
    site: Site, fleet: FleetConfig, curves: SensorCurves
) -> None:
    cm = GeometryCoverage().coverage(site, curves)
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(10):
        t = sample_tactic("blind_spot", site, fleet, curves, rng, coverage=cm)
        hits += sum(cm.score_at(p) == 0.0 for p in t.waypoints[:-1])
    assert hits > 0


def test_perturb_keeps_validity_and_changes_id(
    site: Site, fleet: FleetConfig, curves: SensorCurves
) -> None:
    rng = np.random.default_rng(9)
    for family in FAMILIES:
        parent = sample_tactic(family, site, fleet, curves, rng)
        child = perturb(parent, site, rng)
        assert validate(child, site) == []
        assert child.family == family and child.waypoints[-1] == parent.waypoints[-1]


def test_charge_cycle_ignores_guards(fleet: FleetConfig) -> None:
    chargers = [a for a in fleet.agents if a.charge_time_s > 0]
    assert charge_cycle_s(fleet) == pytest.approx(
        sum(a.endurance_s + a.charge_time_s for a in chargers) / len(chargers)
    )
