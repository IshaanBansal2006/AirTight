from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class Detection:
    """A single sensor detection at one time step.

    Phase 1: camera bounding-box detections.
    Phase 2+: radar returns, lidar clusters — same interface, different obs spaces.
    """

    sensor_id: str
    timestamp: float
    measurement: NDArray[np.float64]  # shape depends on sensor type
    class_label: str | None = None
    class_confidence: float = 0.0
    # Detector-reported attributes beyond class: {"color": "red"} on a vehicle,
    # {"sign_type": "stop"} on a sign. Free-form strings so a
    # new attribute never needs a schema change here.
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class TrackState:
    """Internal state of a single track maintained by the UKF."""

    x: NDArray[np.float64]  # state vector (state_dim,)
    P: NDArray[np.float64]  # covariance matrix (state_dim, state_dim)


@dataclass
class Track:
    """A confirmed track — the output of the tracker.

    `state` is whatever the configured filter carries: `TrackState` (x, P) for
    the standard UKF, `SRTrackState` (x, Cholesky factor) for the SR-UKF
    default. Both expose `.x`.
    """

    track_id: int
    state: TrackState | object  # TrackState | SRTrackState (avoid import cycle)
    # Canonical DS-native evidence state (decision D-B7, 2026-07-29): a mass
    # function over focal SETS of classes — carries the full ignorance structure
    # so next cycle's evidence fuses exactly. Consumers wanting plain numbers
    # call DSClassifier.decide()/pignistic() on it; storing flattened per-class
    # probabilities here would make correct temporal fusion impossible.
    class_beliefs: dict[frozenset[str], float] = field(default_factory=dict)
    age: int = 0  # number of update cycles since initiation
    misses: int = 0  # consecutive update cycles with no associated detection
    source_sensor_ids: list[str] = field(default_factory=list)
    # Schmidt-Kalman consider blocks: per observing drone, the
    # 9x6 cross-covariance between this track's state and that drone's pose
    # error [dp, dtheta]. Carried, never used to update the pose; it is what
    # stops repeated looks from one uncertain camera being counted as
    # independent evidence. Created on first sighting from that drone.
    consider_xc: dict[str, NDArray[np.float64]] = field(default_factory=dict)
