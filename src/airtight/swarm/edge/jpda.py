"""Probabilistic data association.

Ref: Bar-Shalom & Li, "Multitarget-Multisensor Tracking", Ch. 3.

Two association modes share one output convention, so the tracker's downstream
code (combined innovation -> pseudo-measurement update) is identical for both:

  - `compute_association_probabilities`  — PDA. Each track normalises its own
    hypotheses in isolation. EXACT for a single target, and for multi-target
    scenes whose gates never overlap.
  - `compute_joint_association_probabilities` — JPDA. Enumerates FEASIBLE JOINT
    events across all tracks (no detection claimed twice, no track claiming
    two), then marginalises each track's betas out of the joint distribution.

The difference only bites when gates overlap. Independent PDA lets two tracks
each take most of the same detection — physically impossible, since one
detection came from one object — and the result is both filters converging on
the same target, one track starving, and identities swapping when the targets
separate. That is an ID switch, and it is the failure the joint layer exists to
prevent.

Enumeration here is EXHAUSTIVE: exact, but the feasible-event count grows
combinatorially. It is the reference implementation and the exactness oracle a
k-best (Murty) solver is tested against; `max_events` bounds the blow-up and the
overflow is reported, never silently truncated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from airtight.swarm.edge.config import JPDAConfig
    from airtight.swarm.edge.types import Detection

log = logging.getLogger(__name__)

# Gate probability: the chance a true detection falls inside the chi-squared
# 95% gate. Matches the CHI2_95 thresholds used per sensor.
P_GATE = 0.95


@dataclass
class TrackGate:
    """One track's view of this sensor's detections, ready for joint association.

    `gated_idx` holds GLOBAL detection indices (into the sensor's detection
    list) so that different tracks' gates can be compared for conflicts;
    `likelihoods[k]` is N(z_j; z_pred, S) for `gated_idx[k]`.
    """

    gated_idx: list[int]
    likelihoods: list[float] = field(default_factory=list)


def gaussian_likelihood(
    innovation: NDArray[np.float64],
    S: NDArray[np.float64],
) -> float:
    """N(innovation; 0, S) — the measurement likelihood under correct association.

    Uses a linear solve instead of an explicit inverse; the quadratic form is identical.
    """
    m = innovation.shape[0]
    norm = 1.0 / np.sqrt((2.0 * np.pi) ** m * np.linalg.det(S))
    quad = float(innovation @ np.linalg.solve(S, innovation))
    return norm * float(np.exp(-0.5 * quad))


class JPDA:
    def __init__(self, config: JPDAConfig) -> None:
        self.cfg = config
        self.event_overflows = 0  # times enumeration hit max_events (telemetry)

    def gate(
        self,
        detections: list[Detection],
        z_pred: NDArray[np.float64],
        S: NDArray[np.float64],
        gate_threshold: float,
    ) -> list[int]:
        """Mahalanobis gating: indices of detections with d^T S^-1 d < threshold.

        The threshold is chi-squared in the MEASUREMENT dimension (per-sensor,
        from the sensor config), because the innovation is chi-squared with m
        DOF under the correct-association hypothesis.
        """
        if not detections:
            return []
        S_inv = np.linalg.inv(S)
        Z = np.stack([det.measurement for det in detections])
        d = Z - z_pred
        mahal = np.einsum("ni,ij,nj->n", d, S_inv, d)
        return [int(i) for i, ok in enumerate(mahal < gate_threshold) if ok]

    # ------------------------------------------------------------------ PDA
    def compute_association_probabilities(
        self,
        gated_detections: list[Detection],
        z_pred: NDArray[np.float64],
        S: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Association probabilities for ONE track in isolation (parametric PDA).

        betas[0] = P(none of the gated detections is the target's), betas[j] =
        P(detection j is the target's). Each real hypothesis is weighted by
        P_D * N(z_j; z_pred, S) (detected AND looks like the prediction); the
        null hypothesis by lambda * (1 - P_D * P_G) (clutter density times
        "missed or fell outside the gate"). Bayes-normalized.
        """
        weights = [self.cfg.p_false_alarm * (1.0 - self.cfg.p_detection * P_GATE)]
        for det in gated_detections:
            likelihood = gaussian_likelihood(det.measurement - z_pred, S)
            weights.append(self.cfg.p_detection * likelihood)
        betas = np.asarray(weights)
        return betas / betas.sum()

    # ----------------------------------------------------------------- JPDA
    def compute_joint_association_probabilities(
        self,
        gates: list[TrackGate],
    ) -> list[NDArray[np.float64]]:
        """Marginal betas per track, from the JOINT distribution over events.

        Returns one array per track in the SAME layout as the PDA path —
        `out[t][0]` is the null hypothesis and `out[t][k + 1]` corresponds to
        `gates[t].gated_idx[k]` — so `compute_combined_innovation` consumes
        either without knowing which ran.

        Falls back to independent normalisation (and says so) if the feasible
        event count exceeds `max_events`; a k-best solver is the real answer at
        that scale.
        """
        if not gates:
            return []
        events = self._feasible_events(gates)
        if events is None:
            self.event_overflows += 1
            log.warning(
                "joint enumeration exceeded max_events=%d for %d tracks; fell back to "
                "independent PDA for this sensor cycle (betas may double-count a shared "
                "detection). Enable a k-best (Murty) solver to keep this exact.",
                self.cfg.max_events,
                len(gates),
            )
            return [self._independent_betas(g) for g in gates]

        totals = [np.zeros(len(g.gated_idx) + 1) for g in gates]
        for assignment, weight in events:
            for t, k in enumerate(assignment):
                totals[t][k + 1] += weight  # k == -1 lands on index 0, the null slot
        return [
            b / b.sum() if b.sum() > 0 else self._independent_betas(gates[i])
            for i, b in enumerate(totals)
        ]

    def _feasible_events(
        self,
        gates: list[TrackGate],
    ) -> list[tuple[list[int], float]] | None:
        """All feasible joint events with their unnormalised probabilities.

        An event assigns every track either -1 (not detected) or one gated
        detection, such that no detection is claimed twice. Returns None if the
        count would exceed `max_events`.

        Event weight (parametric JPDA):
            prod_{assigned} P_D * N_t(z)  *  prod_{unassigned} (1 - P_D * P_G)
                                          *  lambda ^ (# detections left as clutter)
        Detections outside every gate are clutter in every event, so they
        contribute a constant factor that cancels in the normalisation — only
        the union of the gates is enumerated.
        """
        n_union = len({j for g in gates for j in g.gated_idx})
        miss_w = 1.0 - self.cfg.p_detection * P_GATE
        events: list[tuple[list[int], float]] = []

        def recurse(
            t: int, used: set[int], assignment: list[int], weight: float, n_assigned: int
        ) -> bool:
            """Depth-first over tracks. Returns False if max_events was exceeded."""
            if t == len(gates):
                clutter = n_union - n_assigned
                events.append((list(assignment), weight * self.cfg.p_false_alarm**clutter))
                return len(events) <= self.cfg.max_events
            assignment.append(-1)
            ok = recurse(t + 1, used, assignment, weight * miss_w, n_assigned)
            assignment.pop()
            if not ok:
                return False
            for k, j in enumerate(gates[t].gated_idx):
                if j in used:
                    continue  # the exclusion constraint — this is the whole point
                used.add(j)
                assignment.append(k)
                ok = recurse(
                    t + 1,
                    used,
                    assignment,
                    weight * self.cfg.p_detection * gates[t].likelihoods[k],
                    n_assigned + 1,
                )
                assignment.pop()
                used.discard(j)
                if not ok:
                    return False
            return True

        return events if recurse(0, set(), [], 1.0, 0) else None

    def _independent_betas(self, gate: TrackGate) -> NDArray[np.float64]:
        """PDA normalisation from precomputed likelihoods (the fallback path)."""
        weights = [self.cfg.p_false_alarm * (1.0 - self.cfg.p_detection * P_GATE)]
        weights += [self.cfg.p_detection * lk for lk in gate.likelihoods]
        betas = np.asarray(weights)
        return betas / betas.sum()

    # ---------------------------------------------------------------- shared
    def compute_combined_innovation(
        self,
        gated_detections: list[Detection],
        betas: NDArray[np.float64],
        z_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Probability-weighted innovation: sum_j betas[j] * (z_j - z_pred).

        The null hypothesis (betas[0]) contributes zero innovation, so the
        correction automatically shrinks when the target probably wasn't seen.
        Identical for PDA and JPDA betas — only how the betas were computed
        differs, which is what keeps the joint upgrade a drop-in.
        """
        combined = np.zeros_like(z_pred)
        if gated_detections:
            Z = np.stack([det.measurement for det in gated_detections])
            combined = (betas[1:, None] * (Z - z_pred)).sum(axis=0)
        return combined
