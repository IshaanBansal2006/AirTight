"""Cross-layer message schemas.

Pydantic models are the source of truth; the wire codec is model_dump_json /
model_validate_json (interim transport: JSON in std_msgs/String; upgradeable to
generated .msg/IDL without touching these definitions). No layer owns this
module — L1 produces TrackMsg, L2 consumes intents and produces assignments and
proposals, L3/L4 read everything.

DS mass functions (dict[frozenset[str], float]) are JSON-unfriendly; they cross
the wire with each focal set encoded as a sorted "|"-joined string
("person|vehicle"). encode_mass/decode_mass are the only codec for that.

Uncertainty crosses the wire as a Cholesky FACTOR, not a covariance (040
amendment, 2026-07-30). See TrackMsg.position_sqrt_cov.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def encode_mass(mass: dict[frozenset[str], float]) -> dict[str, float]:
    """DS mass -> JSON-safe dict (focal sets as sorted '|'-joined strings)."""
    return {"|".join(sorted(A)): v for A, v in mass.items()}


def decode_mass(enc: dict[str, float]) -> dict[frozenset[str], float]:
    """Inverse of encode_mass."""
    return {frozenset(k.split("|")): v for k, v in enc.items()}


class TrackMsg(BaseModel):
    """L1 -> L2/L3: one fused track of the world picture."""

    track_id: int
    timestamp: float  # sim seconds
    position: list[float]  # [x, y, z] m, world frame
    velocity: list[float]  # [vx, vy, vz] m/s
    extent: list[float]  # [Lx, Ly, Lz] m (see D-B11 caveats on accuracy)
    # Lower-triangular Cholesky FACTOR L of the marginal position covariance,
    # 3x3 row-major (the three above-diagonal zeros are structural). Recover the
    # covariance as P_pos = L @ L.T. Shipping the factor rather than P is the
    # 040 amendment: a factor cannot round-trip into a non-PSD matrix, and
    # because L is triangular its leading 3x3 block IS the exact factor of the
    # leading 3x3 block of P — so widening this later to 6x6 (position+velocity)
    # or 9x9 is a pure slice change with no re-derivation.
    position_sqrt_cov: list[float]
    class_label: str | None = None  # pignistic decision
    class_confidence: float = 0.0  # BetP of the label
    class_beliefs: dict[str, float] = Field(default_factory=dict)  # encode_mass form
    age: int = 0
    source_sensor_ids: list[str] = Field(default_factory=list)


class TrackFrame(BaseModel):
    """L1 -> L2/L3: ONE COMPLETE confirmed-track picture at one instant.

    The envelope exists so the empty case stays expressive: "L1 is alive and
    currently confirms nothing" is a real, actionable message, and a bare list
    of TrackMsg cannot carry a timestamp when the list is empty. Consumers
    replace their whole picture per frame (snapshot semantics), which is what
    makes track DELETION observable without explicit death messages — the
    deleted track simply isn't in the next frame.
    """

    timestamp: float  # sim seconds — authoritative for the frame
    tracks: list[TrackMsg] = Field(default_factory=list)


class DronePoseMsg(BaseModel):
    """A platform pose. TRUTH from L2's kinematics on
    /drone_poses; L1's ESTIMATE on /drone_pose_estimates. Same shape, so every
    consumer has one code path; the covariance is the only difference."""

    drone_id: str
    timestamp: float  # sim seconds
    position: list[float]  # [x, y, z] m, world frame
    orientation: list[float]  # unit quaternion [w, x, y, z], body -> world
    # Lower-triangular Cholesky FACTOR of the 6x6 pose covariance, row-major,
    # error order [dp (3), dtheta (3)] with dtheta a body-frame rotation vector
    # applied on the right (q [+] dtheta = q (x) exp(dtheta)). None on the
    # truth feed. As with TrackMsg.position_sqrt_cov, the leading 3x3 block is
    # exactly the factor of the position covariance.
    pose_sqrt_cov: list[float] | None = None


class DronePoseFrame(BaseModel):
    """One complete fleet pose picture at one instant (envelope, like TrackFrame)."""

    timestamp: float
    poses: list[DronePoseMsg] = Field(default_factory=list)


class StructuredIntent(BaseModel):
    """Operator -> L2 (Step 1: structured, language-free; Step 3 parses NL into this)."""

    intent_id: str
    verb: Literal["patrol", "track", "scan", "goto"]
    # exactly one target form applies per verb; validation stays permissive at
    # the schema level, the decomposer rejects mismatches with actionable errors
    target_track_id: int | None = None
    area: list[list[float]] | None = None  # polygon [[x, y], ...] world frame
    point: list[float] | None = None  # [x, y, z]
    priority: int = 0
    deadline_s: float | None = None
    drone_whitelist: list[str] | None = None


class TaskAssignment(BaseModel):
    """L2 -> drones: one allocated task."""

    task_id: str
    intent_id: str  # provenance chain back to the operator intent
    drone_id: str
    task_type: str  # primitive task name from the 020 decomposition
    waypoints: list[list[float]] = Field(default_factory=list)  # [[x, y, z], ...]
    target_track_id: int | None = None
    params: dict[str, float] = Field(default_factory=dict)
    priority: int = 0


class EngagementProposal(BaseModel):
    """L2 -> L4 gate: an action requiring explicit human approval."""

    proposal_id: str
    action: str
    target_track_id: int | None = None
    rationale: str
    deadline_s: float  # auto-deny after this many seconds without a decision
