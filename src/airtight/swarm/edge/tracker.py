"""Multi-sensor multi-target tracker — wires filter (014) + PDA (011) + lifecycle.

Rewritten 2026-07-29: consumes the id-keyed sensor dict (D-B1), gates with the
REAL innovation covariance from the filter's unscented measurement prediction
(closing old gap #1), and initiates tracks by two-point differencing of radar
detections (closing old gap #2 — the tracker can now bootstrap from empty).

Per-cycle flow (call step() once per UKFConfig.dt of sim time):
    predict all tracks
    for each sensor with detections:  gate -> PDA betas -> combined innovation
                                      -> pseudo-measurement filter update
    birth: unassociated radar detections paired across consecutive cycles
    lifecycle: hits/misses -> confirm (age >= confirm_hits) / delete (max_misses)

Documented simplifications (see docs/backlog.md):
  - PDA, not joint JPDA: exact until gates overlap (multi-target milestone).
  - Pseudo-measurement update (z_pred + combined innovation) omits PDA's
    covariance-inflation "spread of innovations" term — optimistic in clutter.
  - Camera poses are per-step inputs (`camera_models`): the platform moves, so
    extrinsics cannot live in config. Fixed radar/lidar poses come from config.
    A value may be a bare CameraModel (pose taken as exact — the legacy
    harness) or a MountedCamera: the platform's ESTIMATED pose
    plus its covariance, handled by the Schmidt-Kalman consider update so the
    pose error widens the innovation and is remembered across cycles via the
    track's consider_xc block instead of being re-counted as fresh noise.
  - Classification (decisions 012/015/D-B7): per cycle, each sensor's BEST
    associated class-carrying detection contributes a mass function; the cycle's
    masses fuse via conflict-weighted discounting, then Dempster-combine into
    the track's running DS-native belief (temporal fusion of prior with new
    evidence uses plain Dempster + the K-guard; discounting the prior too is a
    noted alternative).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge.classification import DSClassifier
from airtight.swarm.edge.filters import initial_state, make_filter
from airtight.swarm.edge.jpda import JPDA
from airtight.swarm.edge.observation import CameraModel, h_camera, h_radar
from airtight.swarm.edge.schmidt import ConsiderCamera, MountedCamera, cv_transition
from airtight.swarm.edge.types import Detection, Track

if TYPE_CHECKING:
    from airtight.swarm.edge.config import TrackerConfig

log = logging.getLogger(__name__)

BIRTH_MAX_SPEED = 15.0  # m/s — max plausible target speed for two-point pairing


def _cov(state) -> np.ndarray:
    """P of either state carrier (S S^T for the square-root filter)."""
    S = getattr(state, "S", None)
    return np.asarray(S @ S.T) if S is not None else np.asarray(state.P)


class MultiTargetTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.cfg = config
        self.filter = make_filter(config.ukf)
        self.jpda = JPDA(config.jpda)
        self.classifier = DSClassifier()
        self.tracks: list[Track] = []
        self._next_id: int = 0
        # R per sensor (config stores std-devs; R is the variance diagonal)
        self._R = {
            sid: np.diag(np.square(np.asarray(s.measurement_noise, dtype=float)))
            for sid, s in config.sensors.items()
        }
        # radar detections that matched no track last cycle, awaiting a second
        # sighting for two-point initiation: (position, sim-relative cycle index)
        self._birth_candidates: list[tuple[np.ndarray, int]] = []
        self._cycle = 0
        self._F = cv_transition(config.ukf.dt, config.ukf.state_dim)
        self.consider_repairs = 0
        # detections that fell inside SOME track's gate on the last step() —
        # the SLAMMOT pipeline routes the rest to the ego filters as landmark
        # candidates (a vehicle the tracker has claimed is not a parked car)
        self.claimed: list[Detection] = []
        self._pcc_last: dict[str, np.ndarray] = {}  # per drone, the pose cov last seen

    # ------------------------------------------------------------------ sensors
    def _observation_fn(
        self, sensor_id: str, camera_models: dict[str, CameraModel | MountedCamera]
    ):
        """Bind the sensor's h(x) for this cycle (pose-dependent for cameras)."""
        sensor = self.cfg.sensors[sensor_id]
        if sensor.sensor_type == "camera":
            cam = camera_models.get(sensor_id)
            if not isinstance(cam, CameraModel):
                raise ValueError(
                    f"camera '{sensor_id}' has detections but no CameraModel was "
                    f"passed to step() — camera extrinsics are per-step inputs."
                )
            return lambda x: h_camera(x, cam)
        if sensor.sensor_type == "radar":
            pos = np.asarray(sensor.position or [0.0, 0.0, 0.0])
            return lambda x: h_radar(x, pos)
        raise ValueError(f"unsupported sensor type: {sensor.sensor_type}")

    def _follow_pose_updates(self, camera_models: dict[str, CameraModel | MountedCamera]) -> None:
        """The ego filter corrected its pose since last cycle (sign fixes); carry
        each cross-covariance through that correction from the POSE side:
        P_xc <- P_xc (P_cc_new P_cc_old^-1)^T, the c-side twin of
        _transport_consider. Leaving P_xx alone is conservative (the target
        would also have learned from the sign fix through the correlation), so
        the joint stays PSD; the ego's between-cycle inflation is approximated
        by the same ratio, which is the slowly-varying-consider assumption of 019."""
        for cam in camera_models.values():
            if not isinstance(cam, MountedCamera):
                continue
            did, P_new = cam.pose.drone_id, cam.pose.cov
            P_old = self._pcc_last.get(did)
            if P_old is not None:
                T = np.linalg.solve(P_old.T, P_new.T).T  # P_new @ inv(P_old)
                for track in self.tracks:
                    if did in track.consider_xc:
                        track.consider_xc[did] = track.consider_xc[did] @ T.T
            self._pcc_last[did] = np.array(P_new, copy=True)

    @staticmethod
    def _transport_consider(track: Track, P_before: np.ndarray, skip: str | None = None) -> None:
        """Carry every consider cross-covariance through an update of x.

        P_xc is a covariance BETWEEN x and a drone's pose error; when x is
        corrected by any sensor — the radar, or another drone's camera — the
        correction (I - K H) applies to it too: P_xc+ = (I - K H) P_xc. The
        UT never forms H, but (I - K H) = P_after P_before^-1 exactly in the
        linear-Gaussian case, so the transport needs only the covariances the
        filter already produced. Skipping it leaves the joint
        [[P_xx, P_xc], [P_xc', P_cc]] indefinite within a few radar updates —
        found the hard way (126 factorisation repairs in an 8 s run).
        """
        if not track.consider_xc:
            return
        T = np.linalg.solve(P_before.T, _cov(track.state).T).T  # P_after @ inv(P_before)
        for did in track.consider_xc:
            if did != skip:
                track.consider_xc[did] = T @ track.consider_xc[did]

    @staticmethod
    def _radar_to_position(sensor_pos: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Invert [range, az, el, doppler] to a world position (spherical -> cartesian)."""
        rng, az, el = float(z[0]), float(z[1]), float(z[2])
        horiz = rng * np.cos(el)
        return sensor_pos + np.array([horiz * np.cos(az), horiz * np.sin(az), rng * np.sin(el)])

    # --------------------------------------------------------------------- step
    def step(
        self,
        detections: list[Detection],
        camera_models: dict[str, CameraModel | MountedCamera] | None = None,
    ) -> list[Track]:
        """One tracker cycle. Returns the CONFIRMED tracks."""
        camera_models = camera_models or {}
        self._cycle += 1

        for track in self.tracks:
            track.state = self.filter.predict(track.state)
            for did in track.consider_xc:  # pose error held constant; target moves
                track.consider_xc[did] = self._F @ track.consider_xc[did]
        self._follow_pose_updates(camera_models)

        by_sensor: dict[str, list[Detection]] = {}
        for det in detections:
            by_sensor.setdefault(det.sensor_id, []).append(det)

        associated_tracks: set[int] = set()
        unassociated_radar: list[Detection] = []
        self.claimed = []
        cycle_masses: dict[int, list[dict[frozenset[str], float]]] = {}

        for sensor_id, dets in by_sensor.items():
            sensor = self.cfg.sensors.get(sensor_id)
            if sensor is None:
                log.warning("detections from unknown sensor '%s' ignored", sensor_id)
                continue
            R = self._R[sensor_id]
            mounted = camera_models.get(sensor_id)
            consider = (
                ConsiderCamera(mounted, R, self.cfg.ukf)
                if isinstance(mounted, MountedCamera)
                else None
            )
            h = None if consider else self._observation_fn(sensor_id, camera_models)
            claimed: set[int] = set()  # detection indices gated by ANY track

            for track in self.tracks:
                if consider is not None:
                    P_xc = track.consider_xc.setdefault(
                        mounted.pose.drone_id, np.zeros((self.cfg.ukf.state_dim, 6))
                    )
                    z_pred, S = consider.measurement_prediction(track.state, P_xc)
                else:
                    z_pred, S = self.filter.measurement_prediction(track.state, h, R)
                gated_idx = self.jpda.gate(dets, z_pred, S, sensor.gate_threshold)
                claimed.update(gated_idx)
                if not gated_idx:
                    continue
                gated = [dets[i] for i in gated_idx]
                betas = self.jpda.compute_association_probabilities(gated, z_pred, S)
                combined = self.jpda.compute_combined_innovation(gated, betas, z_pred)
                P_before = _cov(track.state)
                if consider is not None:
                    track.state, P_xc_new, _, _ = consider.update(
                        track.state, P_xc, z_pred + combined
                    )
                    track.consider_xc[mounted.pose.drone_id] = P_xc_new
                    self.consider_repairs += consider.repairs
                    consider.repairs = 0
                    self._transport_consider(track, P_before, skip=mounted.pose.drone_id)
                else:
                    track.state, _, _ = self.filter.update(track.state, z_pred + combined, h, R)
                    self._transport_consider(track, P_before)
                # only count a real association (not null-dominated) as a hit
                if betas[1:].sum() > betas[0]:
                    associated_tracks.add(track.track_id)
                    # classification evidence: this sensor's most probable
                    # associated detection contributes one mass function
                    best = gated[int(np.argmax(betas[1:]))]
                    if best.class_label is not None:
                        cycle_masses.setdefault(track.track_id, []).append(
                            self.classifier.build_mass_function(
                                sensor.sensor_type, best.class_label, best.class_confidence
                            )
                        )

            self.claimed.extend(dets[i] for i in sorted(claimed))
            if sensor.sensor_type == "radar":
                unassociated_radar.extend(d for i, d in enumerate(dets) if i not in claimed)

        # -------- classification fusion (012/015/D-B7) ---------------------
        for track in self.tracks:
            masses = cycle_masses.get(track.track_id)
            if not masses:
                continue
            new = self.classifier.combine_discounted(masses)
            track.class_beliefs = (
                self.classifier.combine(track.class_beliefs, new) if track.class_beliefs else new
            )

        # -------- lifecycle: hits / misses ---------------------------------
        for track in self.tracks:
            if track.track_id in associated_tracks:
                track.misses = 0
                track.age += 1
            else:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses < self.cfg.max_misses]

        # -------- birth: two-point radar initiation ------------------------
        self._initiate(unassociated_radar)

        return [t for t in self.tracks if t.age >= self.cfg.confirm_hits]

    # -------------------------------------------------------------------- birth
    def _initiate(self, unassociated_radar: list[Detection]) -> None:
        """Two-point differencing: pair an unmatched radar detection with one
        from the PREVIOUS cycle to get position + velocity (the init recipe the
        zero-velocity war story mandates — see explanations/edge/ukf.md)."""
        dt = self.cfg.ukf.dt
        new_candidates: list[tuple[np.ndarray, int]] = []
        for det in unassociated_radar:
            sensor = self.cfg.sensors[det.sensor_id]
            pos = self._radar_to_position(np.asarray(sensor.position or [0, 0, 0]), det.measurement)
            paired = False
            for prev_pos, prev_cycle in self._birth_candidates:
                if self._cycle - prev_cycle != 1:
                    continue
                vel = (pos - prev_pos) / dt
                if np.linalg.norm(vel) > BIRTH_MAX_SPEED:
                    continue
                x0 = np.concatenate([pos, vel, self.cfg.birth_extent])  # extent prior
                P0 = np.diag(
                    [0.25] * 3 + [1.0] * 3 + list(np.square(np.asarray(self.cfg.birth_extent_std)))
                )
                self.tracks.append(
                    Track(
                        track_id=self._new_track_id(),
                        state=initial_state(self.filter, x0, P0),
                        age=1,
                    )
                )
                log.info("track %d initiated at %s", self.tracks[-1].track_id, pos.round(2))
                paired = True
                break
            if not paired:
                new_candidates.append((pos, self._cycle))
        self._birth_candidates = new_candidates

    def _new_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid
