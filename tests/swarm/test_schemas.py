"""Round-trip tests for the 040 cross-layer schemas (wire codec = JSON)."""

from __future__ import annotations

import numpy as np

from airtight.swarm.edge.config import UKFConfig
from airtight.swarm.edge.filters import initial_state, make_filter, sqrt_cov_block
from airtight.swarm.schemas import (
    DronePoseFrame,
    DronePoseMsg,
    EngagementProposal,
    StructuredIntent,
    TaskAssignment,
    TrackFrame,
    TrackMsg,
    decode_mass,
    encode_mass,
)


def _track(track_id: int = 0, t: float = 12.3) -> TrackMsg:
    return TrackMsg(
        track_id=track_id,
        timestamp=t,
        position=[1, 2, 0.5],
        velocity=[0.5, 0.2, 0],
        extent=[0.2, 0.2, 0.2],
        position_sqrt_cov=[0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
        class_label="vehicle",
        class_confidence=0.97,
        class_beliefs=encode_mass({frozenset({"vehicle"}): 1.0}),
        age=100,
        source_sensor_ids=["cam_front", "radar_gs"],
    )


def test_mass_codec_roundtrip() -> None:
    mass = {
        frozenset({"vehicle", "person"}): 0.6,
        frozenset({"aircraft"}): 0.1,
        frozenset({"vehicle", "person", "aircraft", "unknown"}): 0.3,
    }
    assert decode_mass(encode_mass(mass)) == mass


def test_schemas_json_roundtrip() -> None:
    msgs = [
        _track(),
        StructuredIntent(intent_id="i1", verb="patrol", area=[[0, 0], [10, 0], [10, 10]]),
        TaskAssignment(
            task_id="t1",
            intent_id="i1",
            drone_id="d1",
            task_type="goto_waypoint",
            waypoints=[[1, 2, 3]],
        ),
        EngagementProposal(
            proposal_id="p1",
            action="designate",
            target_track_id=0,
            rationale="operator intent i1",
            deadline_s=10.0,
        ),
    ]
    for m in msgs:
        assert type(m).model_validate_json(m.model_dump_json()) == m


def test_empty_frame_still_carries_time() -> None:
    """The reason the envelope exists: 'alive, confirming nothing' is a message.

    A bare list[TrackMsg] cannot express this — an empty list has no timestamp,
    so consumers would stall their clocks instead of advancing them.
    """
    frame = TrackFrame(timestamp=42.0, tracks=[])
    assert TrackFrame.model_validate_json(frame.model_dump_json()).timestamp == 42.0


def test_frame_roundtrip_preserves_tracks() -> None:
    frame = TrackFrame(timestamp=9.5, tracks=[_track(0), _track(1)])
    assert TrackFrame.model_validate_json(frame.model_dump_json()) == frame


def test_sqrt_cov_is_a_factor_of_the_leading_block() -> None:
    """L @ L.T must reproduce the marginal position covariance exactly.

    For the SR-UKF this is the free-slice property: S is lower-triangular, so
    S[:3, :3] @ S[:3, :3].T == P[:3, :3] with no re-factorisation.
    """
    P0 = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25])
    P0[0, 3] = P0[3, 0] = 0.5  # position-velocity correlation, deliberately nonzero
    cfg = UKFConfig()
    filt = make_filter(cfg)
    state = initial_state(filt, np.zeros(9), P0)

    L = sqrt_cov_block(state)
    assert np.allclose(L @ L.T, P0[:3, :3])
    assert np.allclose(np.triu(L, k=1), 0.0)  # lower-triangular

    # Widening is a pure slice: the same property must hold at 6x6.
    L6 = sqrt_cov_block(state, dim=6)
    assert np.allclose(L6 @ L6.T, P0[:6, :6])


def test_drone_pose_frame_roundtrip_truth_and_estimate() -> None:
    """041: one schema, covariance optional — truth has none, the estimate does."""
    truth = DronePoseMsg(drone_id="d0", timestamp=1.0, position=[1, 2, 4], orientation=[1, 0, 0, 0])
    L = np.linalg.cholesky(np.diag([0.04, 0.04, 0.09, 1e-4, 1e-4, 4e-4]))
    est = DronePoseMsg(
        drone_id="d0",
        timestamp=1.0,
        position=[1.1, 2.0, 3.9],
        orientation=[0.99, 0, 0, 0.14],
        pose_sqrt_cov=L.ravel().tolist(),
    )
    for frame in (
        DronePoseFrame(timestamp=1.0, poses=[truth]),
        DronePoseFrame(timestamp=1.0, poses=[est]),
        DronePoseFrame(timestamp=2.0),
    ):
        assert DronePoseFrame.model_validate_json(frame.model_dump_json()) == frame
    assert truth.pose_sqrt_cov is None
    # the leading 3x3 block of the 6x6 factor is the factor of the position block
    L3 = np.asarray(est.pose_sqrt_cov).reshape(6, 6)[:3, :3]
    assert np.allclose(L3 @ L3.T, (L @ L.T)[:3, :3])
