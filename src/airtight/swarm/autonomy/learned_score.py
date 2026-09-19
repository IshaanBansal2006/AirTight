"""Localization-aware CBBA path score.

    score(drone, path) = baseline(drone, path) + w . phi(drone, path)

The baseline is the hand-written time-discounted score (diminishing marginal
gain, convergence proof intact). The learned term is a linear ADVANTAGE over
the previous iterate of itself, fitted by approximate policy iteration on
counterfactual credit (training.py). Its features come from an allocation-time
predictor of how well the drone will be localised along the path, built from
the prior sign map — not from running the ego filter, which is what makes the
score cheap enough to evaluate inside CBBA's bundle-building loop.

The predictor is a heuristic: pose variance grows linearly with distance
flown and resets when a leg passes within `sign_range` of a mapped sign. The
learned weights absorb its miscalibration; that is their job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.autonomy.allocator import TimeDiscountedScore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from airtight.swarm.autonomy.decomposer import Task
    from airtight.swarm.autonomy.world_state import DroneState
    from airtight.swarm.scene import Sign

FEATURE_NAMES = (
    "bias",
    "baseline",
    "min_quality",
    "mean_quality",
    "sign_passes",
    "path_length_100m",
    "init_var",
)


def quality(pose_var: float | NDArray[np.float64], s0: float) -> float | NDArray[np.float64]:
    """Localization quality in (0, 1]: 1 at zero variance, 1/2 at s0^2 (the
    return's task weight). Continuous, no threshold to tune."""
    return 1.0 / (1.0 + np.asarray(pose_var) / (s0 * s0))


def _seg_dist(p: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


@dataclass
class LocalizationPredictor:
    """Predicted position-variance trace at each task completion along a path."""

    signs: Sequence[Sign]
    drift_rate: float = 0.02  # m^2 of variance per metre flown (dead reckoning)
    sign_range: float = 12.0  # m — a leg passing this close to a sign gets a fix
    fix_floor: float = 0.01  # m^2 — variance right after a sign fix (0.1 m std)

    def initial_var(self, drone: DroneState) -> float:
        est = drone.pose_estimate
        if est is None or est.pose_sqrt_cov is None:
            return self.fix_floor
        L = np.asarray(est.pose_sqrt_cov).reshape(6, 6)[:3, :3]
        return float(np.trace(L @ L.T))

    def along_path(self, drone: DroneState, path: list[Task]) -> tuple[NDArray[np.float64], int]:
        """(variance at each task's completion, number of sign passes)."""
        var = self.initial_var(drone)
        pos = np.asarray(drone.position, dtype=float)
        out, passes = [], 0
        sign_pos = [np.asarray(s.position, dtype=float) for s in self.signs]
        for task in path:
            wps = [np.asarray(w, dtype=float) for w in task.waypoints] or [pos]
            for wp in wps:
                var += self.drift_rate * float(np.linalg.norm(wp - pos))
                if any(_seg_dist(sp, pos, wp) < self.sign_range for sp in sign_pos):
                    var = self.fix_floor
                    passes += 1
                pos = wp
            out.append(var)
        return np.asarray(out, dtype=float), passes


@dataclass
class LearnedScore:
    predictor: LocalizationPredictor
    weights: NDArray[np.float64]
    s0: float = 1.0
    baseline: TimeDiscountedScore = TimeDiscountedScore()

    def features(self, drone: DroneState, path: list[Task]) -> NDArray[np.float64]:
        base = self.baseline.score(drone, path)
        if not path:
            return np.array([1.0, base, 1.0, 1.0, 0.0, 0.0, self.predictor.initial_var(drone)])
        var, passes = self.predictor.along_path(drone, path)
        q = np.asarray(quality(var, self.s0))
        length = 0.0
        pos = np.asarray(drone.position, dtype=float)
        for task in path:
            for wp in task.waypoints:
                length += float(np.linalg.norm(np.asarray(wp) - pos))
                pos = np.asarray(wp, dtype=float)
        return np.array(
            [
                1.0,
                base,
                float(q.min()),
                float(q.mean()),
                float(passes),
                length / 100.0,
                self.predictor.initial_var(drone),
            ]
        )

    def score(self, drone: DroneState, path: list[Task]) -> float:
        phi = self.features(drone, path)
        return float(phi[1] + self.weights @ phi)

    # ------------------------------------------------------------ persistence
    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "weights": self.weights.tolist(),
                    "features": list(FEATURE_NAMES),
                    "s0": self.s0,
                    "predictor": {
                        "drift_rate": self.predictor.drift_rate,
                        "sign_range": self.predictor.sign_range,
                        "fix_floor": self.predictor.fix_floor,
                    },
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: Path, signs: Sequence[Sign]) -> LearnedScore:
        d = json.loads(path.read_text())
        if list(d["features"]) != list(FEATURE_NAMES):
            raise ValueError(
                f"{path}: feature layout {d['features']} does not match this code's "
                f"{list(FEATURE_NAMES)} — retrain with benchmarks/learned_score.py"
            )
        return cls(
            LocalizationPredictor(signs, **d["predictor"]),
            np.asarray(d["weights"], dtype=float),
            s0=float(d["s0"]),
        )

    @classmethod
    def zero(cls, signs: Sequence[Sign], s0: float = 1.0) -> LearnedScore:
        """Weights all zero: identical to the baseline (the iteration's start)."""
        return cls(LocalizationPredictor(signs), np.zeros(len(FEATURE_NAMES)), s0=s0)
