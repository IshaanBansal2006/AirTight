"""Edge-layer configuration.

Rewritten 2026-07-29 per decisions 013 (9-D state, 3-D radar) and backlog D-B1
(sensors as an id-keyed pydantic discriminated union — matches the
mixed fixed/drone-mounted heterogeneous topology and `Detection.sensor_id`).
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

# chi-squared 95% thresholds by measurement DOF (gate = d^2 < chi2)
CHI2_95 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07, 6: 12.59}


class UKFConfig(BaseModel):
    """UKF tuning parameters (state layout)."""

    state_dim: int = 9  # [x, y, z, vx, vy, vz, Lx, Ly, Lz]

    # Sigma-point scaling. alpha=1, kappa=1 -> lambda=1 -> ALL weights positive
    # at n=9, so covariance reconstructions are PSD by construction. The
    # classic small-alpha default (1e-3) gives center weights ~ +/-1e6 at n=9
    # and drove P indefinite in live runs (see explanations/edge/ukf.md).
    alpha: float = 1.0
    beta: float = 2.0  # prior-distribution knowledge (2.0 = Gaussian-optimal)
    kappa: float = 1.0

    dt: float = 0.1  # prediction time step (s); fusion cadence must match

    # Covariance strategy: "srukf" is the
    # DEFAULT — square-root filter carrying the Cholesky factor (cannot go
    # indefinite; best stress accuracy/consistency). "shortcut" (P - K S K^T)
    # and "joseph" (quadratic-in-K) remain selectable on the standard UKF for
    # comparison. Construct filters via edge.filters.make_filter(), which
    # dispatches on this field.
    covariance_form: Literal["srukf", "shortcut", "joseph"] = "srukf"

    # Process noise — diagonal of Q. Length must equal state_dim.
    # Extent Q = 1e-3 is NEES-tuned (D-B11 sweep, benchmarks/nees_diagnosis.py):
    # rigid bodies don't change size, but tiny Q_ext (1e-8) froze a biased
    # extent belief -> NEES_ext ~9,770 vs ideal 3 ("smug filter"). 1e-3 restores
    # honesty (NEES_ext ~25) and slightly improves accuracy. Here Q compensates
    # observability/model bias, not physical size drift; residual inconsistency
    # is structural (see backlog D-B11 resolution).
    process_noise: list[float] = [1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3]

    @model_validator(mode="after")
    def _q_matches_state(self) -> UKFConfig:
        if len(self.process_noise) != self.state_dim:
            raise ValueError(
                f"process_noise has {len(self.process_noise)} entries; "
                f"state_dim={self.state_dim} — they must match."
            )
        return self


class CameraMount(BaseModel):
    """Where a camera sits on its platform: a body-frame offset
    and a pitch below the platform's forward axis. Yaw 0 = looking along body x."""

    offset: list[float] = [0.0, 0.0, 0.0]  # m, body frame (x fwd, y left, z up)
    pitch_deg: float = 30.0  # positive = nose down
    yaw_deg: float = 0.0


class CameraConfig(BaseModel):
    """A camera sensor: measures pixel bbox [u, v, w, h] (4-DOF).

    A camera either rides a platform (`platform` set: extrinsics come from that
    drone's pose through `mount`) or is posed per step by the
    caller (`platform` None: the legacy chase-camera harness in the tests).
    """

    sensor_type: Literal["camera"] = "camera"
    platform: str | None = None  # drone_id carrying this camera
    mount: CameraMount = CameraMount()
    # Measurement noise — diagonal of R (px^2 as std-devs listed): u, v, w, h.
    measurement_noise: list[float] = [2.0, 2.0, 3.0, 3.0]
    gate_threshold: float = CHI2_95[4]  # 4-DOF measurement
    # Pinhole intrinsics (calibration; extrinsics are per-step platform pose).
    fx: float = 480.0
    fy: float = 480.0
    cx: float = 480.0
    cy: float = 300.0
    width: int = 960
    height: int = 600


class RadarConfig(BaseModel):
    """A 3-D radar: measures [range, azimuth, elevation, doppler] (4-DOF).

    Elevation added — radar alone
    recovers full 3-D position (sensor redundancy / graceful degradation).
    """

    sensor_type: Literal["radar"] = "radar"
    # std-devs: range (m), azimuth (rad), elevation (rad), doppler (m/s)
    measurement_noise: list[float] = [0.1, 0.01, 0.01, 0.1]
    gate_threshold: float = CHI2_95[4]  # 4-DOF measurement
    # Mounting position for FIXED sensors (world frame, m). Drone-mounted
    # radars get their pose per-step from the platform instead.
    position: list[float] | None = None


class LidarConfig(BaseModel):
    """Lidar centroid sensor: [x, y, z] (3-DOF). Phase 3 — stub."""

    sensor_type: Literal["lidar"] = "lidar"
    measurement_noise: list[float] = [0.05, 0.05, 0.05]
    gate_threshold: float = CHI2_95[3]  # 3-DOF measurement


# Discriminated union: pydantic dispatches on sensor_type, so each entry in the
# sensors dict validates against exactly its own schema (D-B1, option A).
SensorConfig = Annotated[
    CameraConfig | RadarConfig | LidarConfig,
    Field(discriminator="sensor_type"),
]


class ImuConfig(BaseModel):
    """IMU error model, as CONTINUOUS-TIME densities so the same
    numbers drive both the synthesiser and the filter's process noise at any
    sample rate. Per-sample std-devs follow from the rate:
        white noise:  sigma * sqrt(rate)      bias random walk:  sigma / sqrt(rate)
    Defaults are consumer-MEMS order of magnitude, not a datasheet.
    """

    rate_hz: float = 100.0
    accel_noise_density: float = 2e-3  # m/s^2 / sqrt(Hz)
    gyro_noise_density: float = 2e-4  # rad/s / sqrt(Hz)
    accel_bias_random_walk: float = 2e-4  # m/s^3 / sqrt(Hz)
    gyro_bias_random_walk: float = 2e-5  # rad/s^2 / sqrt(Hz)
    accel_bias_init_std: float = 0.05  # m/s^2 — turn-on bias spread
    gyro_bias_init_std: float = 0.005  # rad/s
    gravity: list[float] = [0.0, 0.0, -9.81]  # world frame (z up)

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz

    def per_sample(self) -> dict[str, float]:
        """Discrete std-devs at `rate_hz`."""
        r = np.sqrt(self.rate_hz)
        return {
            "accel_noise": self.accel_noise_density * r,
            "gyro_noise": self.gyro_noise_density * r,
            "accel_bias_walk": self.accel_bias_random_walk / r,
            "gyro_bias_walk": self.gyro_bias_random_walk / r,
        }


class EgoConfig(BaseModel):
    """Ego-pose filter: error-state UKF over
    [dp, dv, dtheta, d_accel_bias, d_gyro_bias] + 7 per vehicle landmark."""

    imu: ImuConfig = ImuConfig()
    alpha: float = 1.0  # as UKFConfig: alpha=1, kappa=1 -> every sigma weight positive
    beta: float = 2.0
    kappa: float = 1.0

    # Initial uncertainty (std-devs): the drone knows roughly where it took off
    init_position_std: float = 0.5  # m
    init_velocity_std: float = 0.2  # m/s
    init_attitude_std: float = 0.05  # rad
    # bias init stds come from ImuConfig (the turn-on spread IS the prior)

    # Sign updates against the prior map
    sign_gate: float = CHI2_95[4]
    map_position_std: float = 0.0  # m — prior-map error, folded into R as pixel noise

    # Vehicle landmarks: delayed initialisation by two-view triangulation
    landmark_gate: float = CHI2_95[4]
    parallax_min_deg: float = 3.0  # rays must differ by this much before triangulating
    candidate_min_obs: int = 3
    candidate_min_span_s: float = 1.0  # first-to-last sighting: motion needs time to show
    candidate_max_residual_px: float = 12.0  # static-consistency check on the candidate
    candidate_max_age_s: float = 5.0  # forget a candidate not seen for this long
    vehicle_z_range: list[float] = [0.2, 2.5]  # m — a parked vehicle's centre is near the ground
    duplicate_radius: float = 3.0  # m — a same-colour landmark this close is the same vehicle
    young_landmark_hits: int = 5  # below this, associate on bbox centre only (extent still a prior)
    landmark_probation_misses: int = 20  # predicted fully in view but unmatched -> drop
    vehicle_extent_prior: list[float] = [4.5, 1.85, 1.7]  # m, [L, W, H]
    vehicle_extent_prior_std: list[float] = [0.8, 0.25, 0.4]
    vehicle_yaw_prior_std: float = 0.8  # rad — heading of a parked car is a guess
    bearing_noise_px: float = 2.0  # pixel std used to size a triangulated landmark's cov


class JPDAConfig(BaseModel):
    """JPDA association parameters.

    Gating thresholds live PER-SENSOR now (measurement DOF differs by sensor);
    see each sensor config's `gate_threshold`.
    """

    p_detection: float = 0.9  # probability sensor detects a target in gate
    p_false_alarm: float = 1e-5  # spatial density of false alarms
    # Cap on exhaustively enumerated joint events before falling back to
    # independent PDA (and logging it). Feasible events grow combinatorially in
    # tracks x overlapping detections; a k-best solver replaces enumeration at
    # scale. 10k is generous for the 2-5 target scenes this is exact for.
    max_events: int = 10_000


class TrackerConfig(BaseModel):
    """Top-level tracker config."""

    ukf: UKFConfig = UKFConfig()
    jpda: JPDAConfig = JPDAConfig()

    # Sensors keyed by sensor_id — the same id carried by Detection.sensor_id,
    # so "which model/noise produced this detection" is a single dict lookup.
    sensors: dict[str, SensorConfig] = {
        "cam_front": CameraConfig(),
        "radar_gs": RadarConfig(position=[0.0, 0.0, 0.0]),
    }

    # Birth prior on the 3-D extent: what a new track is assumed
    # to be before any camera has sized it. The default is the small-cube
    # target of the original thin slice; the road scene overrides it with
    # a vehicle-class prior, because a 0.5 m prior against a 4.5 m car puts the
    # first camera bbox ~100 px outside the chi^2 gate and the camera never
    # associates at all.
    birth_extent: list[float] = [0.5, 0.5, 0.5]
    birth_extent_std: list[float] = [0.5, 0.5, 0.5]

    # Track management (M-of-N confirmation)
    confirm_hits: int = 3  # M hits ...
    confirm_window: int = 5  # ... in N scans to confirm
    max_misses: int = 5  # consecutive misses before deletion
