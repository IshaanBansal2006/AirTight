"""Joint JPDA association — the exclusion constraint and its effects."""

from __future__ import annotations

import numpy as np

from airtight.swarm.edge.config import JPDAConfig
from airtight.swarm.edge.jpda import JPDA, TrackGate


def _jpda(**kw) -> JPDA:
    return JPDA(JPDAConfig(**kw))


def test_betas_are_normalised_per_track() -> None:
    j = _jpda()
    gates = [TrackGate([0, 1], [0.8, 0.2]), TrackGate([0, 1], [0.3, 0.7])]
    for b in j.compute_joint_association_probabilities(gates):
        assert np.isclose(b.sum(), 1.0)


def test_disjoint_gates_reduce_to_independent_pda() -> None:
    """No shared detections means no competition — joint must equal independent.

    This is the formal statement of "PDA is exact until gates overlap": the
    two algorithms are the same function on this input.
    """
    j = _jpda()
    gates = [TrackGate([0], [0.6]), TrackGate([1], [0.4])]
    joint = j.compute_joint_association_probabilities(gates)
    for g, b in zip(gates, joint, strict=True):
        assert np.allclose(b, j._independent_betas(g))


def test_shared_detection_is_split_not_double_counted() -> None:
    """The core JPDA property, and the whole reason for the upgrade.

    Two tracks whose ONLY gated detection is the same one. Independent PDA
    hands that detection a high beta to BOTH tracks (each normalises alone).
    Joint enumeration forbids the event where both claim it, so the mass must
    be shared: neither track can hold as much as it would alone.
    """
    j = _jpda()
    shared = [TrackGate([0], [0.5]), TrackGate([0], [0.5])]
    joint = j.compute_joint_association_probabilities(shared)
    independent = [j._independent_betas(g) for g in shared]

    for b_joint, b_indep in zip(joint, independent, strict=True):
        assert b_joint[1] < b_indep[1]  # claim on the shared detection weakened
        assert b_joint[0] > b_indep[0]  # mass moved to "I wasn't detected"


def test_stronger_track_wins_the_contested_detection() -> None:
    """Competition resolves by likelihood ratio, not by track order."""
    j = _jpda()
    gates = [TrackGate([0], [0.9]), TrackGate([0], [0.1])]
    strong, weak = j.compute_joint_association_probabilities(gates)
    assert strong[1] > weak[1]


def test_exclusive_alternative_rescues_a_contested_claim() -> None:
    """Giving one track its own detection frees the shared one for the other.

    Track 0 can only see detection 0. Track 1 sees both. Joint reasoning should
    push track 1 toward detection 1 (its exclusive option) relative to what it
    would claim in isolation, because the events where it takes detection 0 all
    force track 0 to go undetected and are penalised for it.
    """
    j = _jpda()
    gates = [TrackGate([0], [0.8]), TrackGate([0, 1], [0.5, 0.5])]
    joint = j.compute_joint_association_probabilities(gates)
    indep_1 = j._independent_betas(gates[1])
    assert joint[1][2] > indep_1[2]  # detection 1 (exclusive) gains
    assert joint[1][1] < indep_1[1]  # detection 0 (contested) loses


def test_event_overflow_falls_back_loudly() -> None:
    """Blowing the cap must degrade to PDA AND be visible, never silent."""
    j = _jpda(max_events=2)
    gates = [TrackGate([0, 1, 2], [0.3, 0.3, 0.3]) for _ in range(4)]
    out = j.compute_joint_association_probabilities(gates)
    assert j.event_overflows == 1
    assert all(np.isclose(b.sum(), 1.0) for b in out)


def test_empty_and_ungated_inputs_are_safe() -> None:
    j = _jpda()
    assert j.compute_joint_association_probabilities([]) == []
    (only,) = j.compute_joint_association_probabilities([TrackGate([], [])])
    assert np.isclose(only[0], 1.0)  # nothing gated -> certainly not detected
