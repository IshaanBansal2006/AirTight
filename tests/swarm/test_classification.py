"""Tests for Dempster-Shafer classification fusion (decisions 012 + 015).

The centerpiece is the Zadeh-paradox regression: vanilla Dempster must exhibit
the paradox (that's the correct-but-pathological math), and the decision-015
discounting pipeline must NOT.
"""

from __future__ import annotations

from airtight.swarm.edge.classification import CLASSES, DSClassifier

DS = DSClassifier()
THETA = frozenset(CLASSES)


def test_mass_construction_camera_radar_vacuous() -> None:
    cam = DS.build_mass_function("camera", "vehicle", 0.8)
    assert cam[frozenset({"vehicle"})] == 0.8 and abs(cam[THETA] - 0.2) < 1e-12

    rad = DS.build_mass_function("radar", "vehicle", 0.6)
    assert rad[frozenset({"vehicle", "person"})] == 0.6  # broad ground-mover set

    assert DS.build_mass_function("camera", None, 0.9) == {THETA: 1.0}  # no opinion
    for m in (cam, rad):
        assert abs(sum(m.values()) - 1.0) < 1e-12


def test_dempster_combination_reinforces() -> None:
    m1 = DS.build_mass_function("camera", "vehicle", 0.7)
    m2 = DS.build_mass_function("radar", "vehicle", 0.6)
    fused = DS.combine(m1, m2)
    # Broad radar evidence {vehicle, person} cannot raise the SINGLETON belief
    # above the camera's (it doesn't distinguish vehicle from person) — correct
    # DS behavior. What it does do: preserve singleton support, strengthen the
    # broad set, and shrink ignorance.
    assert DS.belief(fused, frozenset({"vehicle"})) >= 0.7 - 1e-12
    assert DS.belief(fused, frozenset({"vehicle", "person"})) > 0.8
    assert fused[frozenset(CLASSES)] < 0.3  # Theta shrank from 0.3/0.4
    assert abs(sum(fused.values()) - 1.0) < 1e-12


def test_zadeh_paradox_vanilla_vs_discounted() -> None:
    """Two confident, contradicting sources + a sliver of shared third class."""
    a = {frozenset({"vehicle"}): 0.99, frozenset({"aircraft"}): 0.01}
    b = {frozenset({"person"}): 0.99, frozenset({"aircraft"}): 0.01}

    # Vanilla Dempster: the paradox — near-certainty in the class NOBODY backed.
    vanilla = DS.combine(a, b)
    assert vanilla[frozenset({"aircraft"})] > 0.99

    # Decision-015 pipeline: heavy mutual conflict -> both discounted -> no
    # absurd certainty; most mass must sit on ignorance, aircraft stays minor.
    fused = DS.combine_discounted([a, b])
    label, conf = DS.decide(fused)
    assert fused.get(THETA, 0.0) > 0.5
    assert DS.pignistic(fused)["aircraft"] < 0.5
    assert conf < 0.5  # no confident call out of a contradiction


def test_belief_plausibility_bounds_and_pignistic() -> None:
    m = DS.combine_discounted(
        [
            DS.build_mass_function("camera", "vehicle", 0.8),
            DS.build_mass_function("radar", "vehicle", 0.5),
        ]
    )
    v = frozenset({"vehicle"})
    bel, pl = DS.belief(m, v), DS.plausibility(m, v)
    betp = DS.pignistic(m)
    assert 0.0 <= bel <= pl <= 1.0
    assert bel <= betp["vehicle"] <= pl  # BetP lives inside the [Bel, Pl] interval
    assert abs(sum(betp.values()) - 1.0) < 1e-9

    label, conf = DS.decide(m)
    assert label == "vehicle" and conf == betp["vehicle"]


def test_discount_limits() -> None:
    m = DS.build_mass_function("camera", "person", 0.9)
    d1 = DS.discount(m, 1.0)  # alpha=1: unchanged (up to float arithmetic)
    for k in m:
        assert abs(d1[k] - m[k]) < 1e-12
    d0 = DS.discount(m, 0.0)  # alpha=0: silenced -> vacuous
    assert abs(d0[THETA] - 1.0) < 1e-12
    assert all(abs(v) < 1e-12 for k, v in d0.items() if k != THETA)
