"""Ego-pose SLAM filter (decisions 018/019): one per drone.

Error-state unscented Kalman filter. The NOMINAL state lives outside the
filter — position, velocity, attitude quaternion, IMU biases, and one block of
[position, extent, yaw] per vehicle landmark — and the filter estimates a
small ERROR around it:

    delta = [dp(3), dv(3), dtheta(3), dba(3), dbg(3) | dp_l(3), dL_l(3), dyaw_l(1) ...]

with dtheta a body-frame rotation vector applied on the right (rotation.py).
Sigma points are drawn on the error, applied to the nominal by the retraction,
pushed through the IMU integration or the camera model, and pulled back to
error vectors around the propagated nominal. That is what lets a quaternion
sit in the state without a rank-deficient covariance, and what lets the
landmark measurement model be `h_camera_box` at the ego pose with no
Jacobian (the 010 reasoning, reused).

Predict: strapdown integration of each IMU sample (imu.integrate) with the
bias sigma points subtracted; process noise from ImuConfig, per sample.

Update, per camera frame:
  signs     -> known position/size from the prior map; association by
               type + chi^2 gate on the unscented innovation; a sighting is a
               full position fix (standard size gives range).
  vehicles  -> associated to in-state landmarks by colour + gate; leftovers
               become CANDIDATES. A candidate is promoted only after being
               seen from two viewpoints with enough parallax (delayed
               initialisation), triangulated, and checked for consistency
               with a STATIC point — a moving vehicle fails that check. That
               consistency check is the SLAMMOT discrimination: the filter
               never trusts the class label to say "landmark".

Documented approximations (see explanations/edge/ego.md):
  - covariance form with a PSD repair, not square-root (state size changes
    every time a landmark is added);
  - a new landmark's cross-covariance with the pose is zero at init; its
    marginal is inflated by the pose covariance instead;
  - landmark maps are per drone, never shared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import CHI2_95, CameraConfig, EgoConfig
from airtight.swarm.edge.imu import ImuSample, integrate
from airtight.swarm.edge.observation import _CORNER_SIGNS, CameraModel, box_corners, in_view_box
from airtight.swarm.edge.sensing import mount_rotation
from airtight.swarm.scene import SIGN_EXTENTS, Sign
from airtight.swarm.schemas import DronePoseMsg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from airtight.swarm.edge.types import Detection

log = logging.getLogger(__name__)

POSE_DIM = 15  # [dp, dv, dtheta, dba, dbg]
LM_DIM = 7  # [dp_l, dL_l, dyaw_l]
_POSE_SLICE = np.r_[0:3, 6:9]  # the 6-D [dp, dtheta] block the wire and the tracker want


@dataclass
class Landmark:
    """A vehicle landmark in the state (nominal part; its error block lives in P)."""

    landmark_id: int
    color: str
    position: NDArray[np.float64]
    extent: NDArray[np.float64]
    yaw: float
    hits: int = 0
    misses: int = 0  # consecutive frames predicted in view but unmatched


@dataclass
class _Sighting:
    t: float
    cam: CameraModel
    origin: NDArray[np.float64]
    ray: NDArray[np.float64]  # unit, world frame
    bbox: NDArray[np.float64]


@dataclass
class _Candidate:
    color: str
    sightings: list[_Sighting] = field(default_factory=list)


@dataclass
class UpdateReport:
    signs_used: int = 0
    landmarks_used: int = 0
    landmarks_added: list[int] = field(default_factory=list)
    landmarks_dropped: list[int] = field(default_factory=list)
    candidates_rejected: int = 0
    unassociated: list[Detection] = field(default_factory=list)


class EgoFilter:
    def __init__(
        self,
        config: EgoConfig,
        signs: Sequence[Sign],
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        orientation: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.cfg = config
        self.signs = list(signs)
        self.p = np.asarray(position, dtype=float).copy()
        self.v = np.asarray(velocity, dtype=float).copy()
        self.q = rotation.normalize(np.asarray(orientation, dtype=float))
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.landmarks: list[Landmark] = []
        self._candidates: dict[str, _Candidate] = {}
        self._next_lm_id = 0
        self.repairs = 0
        self.rng = rng if rng is not None else np.random.default_rng(0)

        imu = config.imu
        self.P = np.diag(
            np.square(
                np.array(
                    [config.init_position_std] * 3
                    + [config.init_velocity_std] * 3
                    + [config.init_attitude_std] * 3
                    + [imu.accel_bias_init_std] * 3
                    + [imu.gyro_bias_init_std] * 3
                )
            )
        )
        n = imu.per_sample()
        dt = imu.dt
        # discrete process noise on the error state per IMU sample (Sola 2017 §5.4)
        self._Q_pose = np.diag(
            np.array(
                [0.25 * (n["accel_noise"] * dt * dt) ** 2] * 3  # dp: second integral
                + [(n["accel_noise"] * dt) ** 2] * 3  # dv
                + [(n["gyro_noise"] * dt) ** 2] * 3  # dtheta
                + [n["accel_bias_walk"] ** 2] * 3
                + [n["gyro_bias_walk"] ** 2] * 3
            )
        )
        self._gravity = np.asarray(imu.gravity, dtype=float)

    # ---------------------------------------------------------------- shape
    @property
    def n(self) -> int:
        return POSE_DIM + LM_DIM * len(self.landmarks)

    def _weights(self, n: int) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        lam = self.cfg.alpha**2 * (n + self.cfg.kappa) - n
        Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wc = Wm.copy()
        Wm[0] = lam / (n + lam)
        Wc[0] = Wm[0] + (1.0 - self.cfg.alpha**2 + self.cfg.beta)
        return float(np.sqrt(n + lam)), Wm, Wc

    def _sigma_errors(self) -> NDArray[np.float64]:
        """(2n+1, n) error-space sigma points around zero: 0, +gamma*cols(S), -gamma*cols(S)."""
        n = self.n
        gamma, _, _ = self._weights(n)
        S = self._factor(self.P)
        offsets = gamma * S.T  # rows are the column vectors of S
        return np.vstack([np.zeros((1, n)), offsets, -offsets])

    def _factor(self, P: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            return np.asarray(np.linalg.cholesky(P), dtype=float)
        except np.linalg.LinAlgError:
            self.repairs += 1
            w, V = np.linalg.eigh(0.5 * (P + P.T))
            floor = 1e-12 * max(float(w[-1]), 1.0)
            self.P = (V * np.clip(w, floor, None)) @ V.T
            return np.asarray(np.linalg.cholesky(self.P), dtype=float)

    # ------------------------------------------------------------ retraction
    def _apply(self, deltas: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Nominal [+] delta for a batch of error vectors -> batched nominal states."""
        m = deltas.shape[0]
        out = {
            "p": self.p + deltas[:, 0:3],
            "v": self.v + deltas[:, 3:6],
            "q": rotation.boxplus(np.broadcast_to(self.q, (m, 4)), deltas[:, 6:9]),
            "ba": self.ba + deltas[:, 9:12],
            "bg": self.bg + deltas[:, 12:15],
        }
        for k, lm in enumerate(self.landmarks):
            o = POSE_DIM + LM_DIM * k
            out[f"lp{k}"] = lm.position + deltas[:, o : o + 3]
            out[f"lL{k}"] = lm.extent + deltas[:, o + 3 : o + 6]
            out[f"ly{k}"] = lm.yaw + deltas[:, o + 6]
        return out

    def _retract_mean(self, delta: NDArray[np.float64]) -> None:
        """Fold a mean error into the nominal state (after an update)."""
        self.p = self.p + delta[0:3]
        self.v = self.v + delta[3:6]
        self.q = rotation.boxplus(self.q, delta[6:9])
        self.ba = self.ba + delta[9:12]
        self.bg = self.bg + delta[12:15]
        for k, lm in enumerate(self.landmarks):
            o = POSE_DIM + LM_DIM * k
            lm.position = lm.position + delta[o : o + 3]
            lm.extent = lm.extent + delta[o + 3 : o + 6]
            lm.yaw = float(rotation.wrap_angle(lm.yaw + delta[o + 6]))

    # --------------------------------------------------------------- predict
    def predict(self, imu: ImuSample) -> None:
        n = self.n
        _, Wm, Wc = self._weights(n)
        deltas = self._sigma_errors()
        st = self._apply(deltas)
        dt = self.cfg.imu.dt
        p1, v1, q1 = integrate(
            st["p"], st["v"], st["q"], imu.accel - st["ba"], imu.gyro - st["bg"], dt, self._gravity
        )
        # new nominal = the propagated centre point; errors of the others around it
        self.p, self.v, self.q = p1[0].copy(), v1[0].copy(), q1[0].copy()
        errs = np.zeros_like(deltas)
        errs[:, 0:3] = p1 - self.p
        errs[:, 3:6] = v1 - self.v
        errs[:, 6:9] = rotation.boxminus(q1, np.broadcast_to(self.q, q1.shape))
        errs[:, 9:] = deltas[:, 9:]  # biases and landmarks are static in the model
        mean = Wm @ errs
        dev = errs - mean
        P = (dev.T * Wc) @ dev
        P[:POSE_DIM, :POSE_DIM] += self._Q_pose
        self.P = 0.5 * (P + P.T)
        self._retract_mean(mean)

    # ---------------------------------------------------------------- camera
    def _camera_at(
        self,
        p: NDArray[np.float64],
        q: NDArray[np.float64],
        R_bc: NDArray[np.float64],
        cfg: CameraConfig,
    ) -> CameraModel:
        R_wb = rotation.to_matrix(q)
        return CameraModel(
            fx=cfg.fx,
            fy=cfg.fy,
            cx=cfg.cx,
            cy=cfg.cy,
            width=cfg.width,
            height=cfg.height,
            R_wc=(R_wb @ R_bc).T,
            t_w=p + R_wb @ np.asarray(cfg.mount.offset),
        )

    def _unscented_box(
        self,
        deltas: NDArray[np.float64],
        st: dict[str, NDArray[np.float64]],
        R_bc: NDArray[np.float64],
        cfg: CameraConfig,
        center: NDArray[np.float64] | None,
        extent: NDArray[np.float64] | None,
        yaw: float | None,
        lm_index: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Predicted bbox, its covariance (without R) and the error/measurement
        cross-covariance for a box that is either fixed (sign) or landmark k.

        Vectorised over the sigma set: one camera per sigma point, all boxes
        projected in a single einsum. The per-point Python loop this replaced
        was 90% of a training rollout (profiled 2026-09-13)."""
        m = deltas.shape[0]
        R_wb = rotation.to_matrix(st["q"])  # (m, 3, 3)
        R_wc = np.transpose(R_wb @ R_bc, (0, 2, 1))  # world -> camera, per sigma
        t_w = st["p"] + R_wb @ np.asarray(cfg.mount.offset, dtype=float)  # (m, 3)
        if lm_index is None:
            corners_w = np.broadcast_to(box_corners(center, extent, yaw), (m, 8, 3))  # type: ignore[arg-type]
        else:
            corners_w = _box_corners_batch(
                st[f"lp{lm_index}"], st[f"lL{lm_index}"], st[f"ly{lm_index}"]
            )
        corners_c = np.einsum("mij,mkj->mki", R_wc, corners_w - t_w[:, None, :])  # (m, 8, 3)
        z = corners_c[:, :, 2]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = cfg.fx * corners_c[:, :, 0] / z + cfg.cx
        v = cfg.fy * corners_c[:, :, 1] / z + cfg.cy
        u_min, u_max, v_min, v_max = u.min(1), u.max(1), v.min(1), v.max(1)
        Z = np.stack(
            [0.5 * (u_min + u_max), 0.5 * (v_min + v_max), u_max - u_min, v_max - v_min], 1
        )
        _, Wm, Wc = self._weights(self.n)
        z_bar = Wm @ Z
        dz = Z - z_bar
        Pzz = (dz.T * Wc) @ dz
        Pxz = (deltas.T * Wc) @ dz
        return z_bar, Pzz, Pxz

    @staticmethod
    def _centre_visible(center: NDArray[np.float64], cam: CameraModel) -> bool:
        """Association pre-check: the box CENTRE projects inside the image. Looser
        than in_view_box on purpose — a young landmark's extent and yaw are still
        priors, so its predicted box can straddle the image edge while the real
        one is comfortably inside; the chi^2 gate does the real deciding."""
        p_c = cam.R_wc @ (np.asarray(center, dtype=float) - cam.t_w)
        if p_c[2] <= 0.0:
            return False
        u = cam.fx * p_c[0] / p_c[2] + cam.cx
        v = cam.fy * p_c[1] / p_c[2] + cam.cy
        return bool(0.0 <= u <= cam.width and 0.0 <= v <= cam.height)

    def _kalman(
        self,
        z: NDArray[np.float64],
        z_bar: NDArray[np.float64],
        Pzz: NDArray[np.float64],
        Pxz: NDArray[np.float64],
    ) -> None:
        K = np.linalg.solve(Pzz, Pxz.T).T
        self._retract_mean(K @ (z - z_bar))
        P = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P + P.T)

    def update(
        self, detections: list[Detection], cam_cfg: CameraConfig, t: float, R: NDArray[np.float64]
    ) -> UpdateReport:
        """One camera frame of landmark-eligible detections (signs + vehicles
        nobody else claimed). Sequential updates; report says what happened."""
        report = UpdateReport()
        R_bc = mount_rotation(cam_cfg.mount)
        signs = [d for d in detections if d.class_label == "sign"]
        vehicles = [d for d in detections if d.class_label == "vehicle"]

        for d in signs:
            if not self._update_sign(d, cam_cfg, R_bc, R):
                report.unassociated.append(d)
            else:
                report.signs_used += 1

        matched_lm: set[int] = set()
        leftovers: list[Detection] = []
        for d in vehicles:
            k = self._update_landmark(d, cam_cfg, R_bc, R, matched_lm)
            if k is None:
                leftovers.append(d)
            else:
                matched_lm.add(k)
                report.landmarks_used += 1

        self._probation(cam_cfg, R_bc, matched_lm, report)
        cam_nominal = self._camera_at(self.p, self.q, R_bc, cam_cfg)
        for d in leftovers:
            self._candidate(d, cam_nominal, t, report)
        self._expire_candidates(t)
        report.unassociated.extend(leftovers)
        return report

    # ----------------------------------------------------------------- signs
    def _sign_R(
        self, R: NDArray[np.float64], rng: float, cam_cfg: CameraConfig
    ) -> NDArray[np.float64]:
        """Map error enters as pixel error: sigma_map metres at range r is fx*sigma/r px."""
        if self.cfg.map_position_std <= 0.0:
            return R
        px = cam_cfg.fx * self.cfg.map_position_std / max(rng, 1e-3)
        return R + np.diag([px * px, px * px, 0.0, 0.0])

    def _update_sign(
        self, d: Detection, cam_cfg: CameraConfig, R_bc: NDArray[np.float64], R: NDArray[np.float64]
    ) -> bool:
        kind = d.attributes.get("sign_type")
        cam0 = self._camera_at(self.p, self.q, R_bc, cam_cfg)
        deltas = self._sigma_errors()
        st = self._apply(deltas)
        best: tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None = (
            None
        )
        for sign in self.signs:
            if kind is not None and sign.sign_type != kind:
                continue
            center = np.asarray(sign.position, dtype=float)
            if not self._centre_visible(center, cam0):
                continue
            z_bar, Pzz, Pxz = self._unscented_box(
                deltas, st, R_bc, cam_cfg, center, sign.extent, sign.yaw, None
            )
            S = Pzz + self._sign_R(R, float(np.linalg.norm(center - cam0.t_w)), cam_cfg)
            nu = d.measurement - z_bar
            d2 = float(nu @ np.linalg.solve(S, nu))
            if d2 < self.cfg.sign_gate and (best is None or d2 < best[0]):
                best = (d2, z_bar, S, Pxz)
        if best is None:
            return False
        _, z_bar, S, Pxz = best
        self._kalman(d.measurement, z_bar, S, Pxz)
        return True

    # ------------------------------------------------------------- landmarks
    def _update_landmark(
        self,
        d: Detection,
        cam_cfg: CameraConfig,
        R_bc: NDArray[np.float64],
        R: NDArray[np.float64],
        taken: set[int],
    ) -> int | None:
        if not self.landmarks:
            return None
        color = d.attributes.get("color")
        cam0 = self._camera_at(self.p, self.q, R_bc, cam_cfg)
        deltas = self._sigma_errors()
        st = self._apply(deltas)
        best: (
            tuple[float, int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
        ) = None
        for k, lm in enumerate(self.landmarks):
            if k in taken or (color is not None and lm.color != color):
                continue
            if not self._centre_visible(lm.position, cam0):
                continue
            z_bar, Pzz, Pxz = self._unscented_box(deltas, st, R_bc, cam_cfg, None, None, None, k)
            S = Pzz + R
            nu = d.measurement - z_bar
            if lm.hits < self.cfg.young_landmark_hits:
                # extent/yaw are still the class prior: judge the match on where
                # the box is, not how big it is (2-DOF gate on the centre)
                d2 = float(nu[:2] @ np.linalg.solve(S[:2, :2], nu[:2]))
                gate = CHI2_95[2]
            else:
                d2 = float(nu @ np.linalg.solve(S, nu))
                gate = self.cfg.landmark_gate
            if d2 < gate and (best is None or d2 < best[0]):
                best = (d2, k, z_bar, S, Pxz)
        if best is None:
            return None
        _, k, z_bar, S, Pxz = best
        self._kalman(d.measurement, z_bar, S, Pxz)
        self.landmarks[k].hits += 1
        self.landmarks[k].misses = 0
        return k

    def _probation(
        self,
        cam_cfg: CameraConfig,
        R_bc: NDArray[np.float64],
        matched: set[int],
        report: UpdateReport,
    ) -> None:
        """A landmark predicted in view but unmatched too many times in a row was
        probably a vehicle that drove off (or a bad triangulation): drop it."""
        cam0 = self._camera_at(self.p, self.q, R_bc, cam_cfg)
        keep: list[int] = []
        for k, lm in enumerate(self.landmarks):
            # strict visibility here: the synthesiser/detector reports nothing
            # for a box cut off by the image edge, and that silence is not a miss
            if k not in matched and in_view_box(lm.position, lm.extent, lm.yaw, cam0):
                lm.misses += 1
            if lm.misses >= self.cfg.landmark_probation_misses:
                report.landmarks_dropped.append(lm.landmark_id)
            else:
                keep.append(k)
        if len(keep) != len(self.landmarks):
            idx = list(range(POSE_DIM)) + [
                POSE_DIM + LM_DIM * k + j for k in keep for j in range(LM_DIM)
            ]
            self.P = self.P[np.ix_(idx, idx)]
            self.landmarks = [self.landmarks[k] for k in keep]

    # ------------------------------------------------- delayed initialisation
    def _candidate(self, d: Detection, cam: CameraModel, t: float, report: UpdateReport) -> None:
        color = d.attributes.get("color", "?")
        u, v = float(d.measurement[0]), float(d.measurement[1])
        ray_c = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
        ray_w = cam.R_wc.T @ ray_c
        ray_w /= np.linalg.norm(ray_w)
        cand = self._candidates.setdefault(color, _Candidate(color=color))
        cand.sightings.append(_Sighting(t, cam, cam.t_w.copy(), ray_w, d.measurement.copy()))
        if len(cand.sightings) < self.cfg.candidate_min_obs:
            return
        first, last = cand.sightings[0], cand.sightings[-1]
        parallax = float(np.arccos(np.clip(first.ray @ last.ray, -1.0, 1.0)))
        if (
            parallax < np.deg2rad(self.cfg.parallax_min_deg)
            or last.t - first.t < self.cfg.candidate_min_span_s
        ):
            return
        point = _triangulate(first.origin, first.ray, last.origin, last.ray)
        z_lo, z_hi = self.cfg.vehicle_z_range
        if point is None or not (z_lo <= point[2] <= z_hi):
            # enough parallax, yet the rays meet behind a camera, nowhere, or
            # off the ground plane: no parked vehicle explains both sightings.
            # (Radial motion along the line of sight is the weak case for a
            # bearings-only check; the radar-initiated tracker is the primary
            # defence against mapping the moving target — see the pipeline.)
            report.candidates_rejected += 1
            del self._candidates[color]
            return
        for lm in self.landmarks:
            if (
                lm.color == color
                and np.linalg.norm(lm.position - point) < self.cfg.duplicate_radius
            ):
                del self._candidates[color]  # already mapped; association will pick it up
                return
        # static-consistency: every sighting must look at THIS point
        for s in cand.sightings:
            p_c = s.cam.R_wc @ (point - s.cam.t_w)
            if p_c[2] <= 0.0:
                resid = np.inf
            else:
                resid = float(
                    np.hypot(
                        s.cam.fx * p_c[0] / p_c[2] + s.cam.cx - s.bbox[0],
                        s.cam.fy * p_c[1] / p_c[2] + s.cam.cy - s.bbox[1],
                    )
                )
            if resid > self.cfg.candidate_max_residual_px:
                report.candidates_rejected += 1
                del self._candidates[color]
                return
        self._add_landmark(color, point, first, last, parallax, report)
        del self._candidates[color]

    def _add_landmark(
        self,
        color: str,
        point: NDArray[np.float64],
        first: _Sighting,
        last: _Sighting,
        parallax: float,
        report: UpdateReport,
    ) -> None:
        rng = float(np.linalg.norm(point - last.origin))
        sigma_bearing = self.cfg.bearing_noise_px / last.cam.fx
        across = rng * sigma_bearing
        along = across / max(np.sin(parallax), 1e-3)
        ray = last.ray
        P_pos = (along**2 - across**2) * np.outer(ray, ray) + across**2 * np.eye(3)
        P_pos += self.P[0:3, 0:3]  # the pose was uncertain when we looked (no cross term kept)
        P_lm = np.zeros((LM_DIM, LM_DIM))
        P_lm[0:3, 0:3] = P_pos
        P_lm[3:6, 3:6] = np.diag(np.square(np.asarray(self.cfg.vehicle_extent_prior_std)))
        P_lm[6, 6] = self.cfg.vehicle_yaw_prior_std**2
        n0 = self.n
        P = np.zeros((n0 + LM_DIM, n0 + LM_DIM))
        P[:n0, :n0] = self.P
        P[n0:, n0:] = P_lm
        self.P = P
        lm = Landmark(
            self._next_lm_id,
            color,
            point.copy(),
            np.asarray(self.cfg.vehicle_extent_prior, dtype=float).copy(),
            0.0,
            hits=1,
        )
        self._next_lm_id += 1
        self.landmarks.append(lm)
        report.landmarks_added.append(lm.landmark_id)
        log.info(
            "landmark %d (%s) initialised at %s from %.1f deg parallax",
            lm.landmark_id,
            color,
            point.round(2),
            np.rad2deg(parallax),
        )

    def _expire_candidates(self, t: float) -> None:
        for color in list(self._candidates):
            if t - self._candidates[color].sightings[-1].t > self.cfg.candidate_max_age_s:
                del self._candidates[color]

    # --------------------------------------------------------------- outputs
    def pose_cov(self) -> NDArray[np.float64]:
        """6x6 covariance of [dp, dtheta] — the block the wire and the consider update use."""
        return np.asarray(self.P[np.ix_(_POSE_SLICE, _POSE_SLICE)])

    def pose_sqrt_cov(self) -> NDArray[np.float64]:
        return self._factor_of(self.pose_cov())

    def _factor_of(self, P: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            return np.asarray(np.linalg.cholesky(P), dtype=float)
        except np.linalg.LinAlgError:
            self.repairs += 1
            w, V = np.linalg.eigh(0.5 * (P + P.T))
            return np.asarray(np.linalg.cholesky((V * np.clip(w, 1e-12, None)) @ V.T), dtype=float)

    def to_msg(self, drone_id: str, t: float) -> DronePoseMsg:
        return DronePoseMsg(
            drone_id=drone_id,
            timestamp=t,
            position=self.p.tolist(),
            orientation=self.q.tolist(),
            pose_sqrt_cov=self.pose_sqrt_cov().ravel().tolist(),
        )

    def landmark_cov(self, k: int) -> NDArray[np.float64]:
        o = POSE_DIM + LM_DIM * k
        return np.asarray(self.P[o : o + LM_DIM, o : o + LM_DIM])


def _box_corners_batch(
    center: NDArray[np.float64], extent: NDArray[np.float64], yaw: NDArray[np.float64]
) -> NDArray[np.float64]:
    """box_corners for a batch: centres (m,3), extents (m,3), yaws (m,) -> (m, 8, 3)."""
    c, s = np.cos(yaw), np.sin(yaw)
    zeros, ones = np.zeros_like(c), np.ones_like(c)
    Rz = np.stack(
        [
            np.stack([c, -s, zeros], -1),
            np.stack([s, c, zeros], -1),
            np.stack([zeros, zeros, ones], -1),
        ],
        -2,
    )  # (m, 3, 3)
    local = _CORNER_SIGNS[None, :, :] * (extent[:, None, :] / 2.0)  # (m, 8, 3)
    return np.asarray(np.einsum("mij,mkj->mki", Rz, local) + center[:, None, :], dtype=float)


def _triangulate(
    o1: NDArray[np.float64],
    r1: NDArray[np.float64],
    o2: NDArray[np.float64],
    r2: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Midpoint of the closest approach of two rays; None if (near) parallel or behind."""
    w = o1 - o2
    b = float(r1 @ r2)
    d, e = float(r1 @ w), float(r2 @ w)
    denom = 1.0 - b * b
    if denom < 1e-9:
        return None
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    if s <= 0.0 or t <= 0.0:
        return None
    return np.asarray(0.5 * ((o1 + s * r1) + (o2 + t * r2)))


def sign_extent(sign_type: str) -> NDArray[np.float64]:
    return np.asarray(SIGN_EXTENTS[sign_type], dtype=float)
