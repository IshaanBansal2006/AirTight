"""Dempster-Shafer classification fusion (decisions 012 + 015).

Conflict handling: conflict-weighted Shafer discounting (each source loses
voting power in proportion to its mean pairwise conflict with the others),
then standard Dempster combination. Decision rule: pignistic transform (BetP).

Refs: Shafer (1976); Smets' transferable-belief model (pignistic level);
decision docs 012 and 015 for the option space and rationale.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

CLASSES: list[str] = ["vehicle", "person", "aircraft", "unknown"]

# Broad focal elements for coarse sensors. Keys are the labels a radar-side
# classifier may emit; values are the focal sets the evidence actually supports.
_RADAR_BROAD: dict[str, frozenset[str]] = {
    "vehicle": frozenset({"vehicle", "person"}),  # ground mover
    "person": frozenset({"vehicle", "person"}),
    "aircraft": frozenset({"aircraft"}),
}

_NEAR_TOTAL_CONFLICT = 1.0 - 1e-9

Mass = dict[frozenset[str], float]


class DSClassifier:
    def __init__(self, classes: list[str] | None = None) -> None:
        self.classes = classes or CLASSES
        self.theta: frozenset[str] = frozenset(self.classes)

    # ---------------------------------------------------------- mass building
    def build_mass_function(
        self,
        sensor_type: str,
        class_label: str | None,
        confidence: float,
    ) -> Mass:
        """Mass function from one sensor classification.

        Camera/lidar commit `confidence` to the singleton {label}; radar commits
        to its broad category. The remainder goes to Theta — *unmodeled
        ignorance*, deliberately NOT to the class "unknown" (Theta = "could be
        any class" is different from the specific label "unknown").
        No label at all -> the vacuous mass (all on Theta): "no opinion".
        """
        if class_label is None or class_label not in self.theta:
            return {self.theta: 1.0}
        conf = min(max(float(confidence), 0.0), 1.0)
        if sensor_type == "radar":
            focal = _RADAR_BROAD.get(class_label, self.theta)
        else:  # camera, lidar: fine-grained singleton evidence
            focal = frozenset({class_label})
        if focal == self.theta or conf == 0.0:
            return {self.theta: 1.0}
        return {focal: conf, self.theta: 1.0 - conf}

    # ------------------------------------------------------ conflict/discount
    def conflict(self, m1: Mass, m2: Mass) -> float:
        """Dempster conflict K: total mass landing on empty intersections."""
        return sum(v1 * v2 for B, v1 in m1.items() for C, v2 in m2.items() if not (B & C))

    def discount(self, mass: Mass, alpha: float) -> Mass:
        """Shafer discounting m^alpha: scale committed mass by alpha, dump the
        rest on Theta. alpha=1 -> unchanged; alpha=0 -> vacuous (no vote)."""
        out: Mass = {}
        for A, v in mass.items():
            if self.theta != A:
                out[A] = out.get(A, 0.0) + alpha * v
        out[self.theta] = out.get(self.theta, 0.0) + 1.0 - alpha * (1.0 - mass.get(self.theta, 0.0))
        return out

    def combine(self, m1: Mass, m2: Mass) -> Mass:
        """Dempster's rule: intersect focal sets, renormalize conflict away.

        Near-total conflict (K -> 1) makes 1/(1-K) explode — the Zadeh regime.
        Guarded: returns the vacuous mass and logs, rather than emitting the
        paradoxical certainty. The discounting pipeline exists precisely to
        keep inputs out of this regime.
        """
        out: Mass = {}
        conflict = 0.0
        for B, v1 in m1.items():
            for C, v2 in m2.items():
                inter = B & C
                if inter:
                    out[inter] = out.get(inter, 0.0) + v1 * v2
                else:
                    conflict += v1 * v2
        if conflict >= _NEAR_TOTAL_CONFLICT:
            log.warning(
                "Dempster combination: near-total conflict (K=%.6f); returning vacuous mass",
                conflict,
            )
            return {self.theta: 1.0}
        scale = 1.0 / (1.0 - conflict)
        return {A: v * scale for A, v in out.items()}

    def combine_discounted(self, masses: list[Mass]) -> Mass:
        """Decision-015 pipeline: conflict-weighted discounting, then Dempster.

        Source i is discounted by alpha_i = 1 - mean pairwise conflict with the
        other sources: a source that disagrees with everyone loses voting power
        (reliability semantics), instead of global ignorance (Yager) or
        evidence averaging (Murphy). Once discounted, Dempster's rule keeps its
        associativity, so the fold order is irrelevant.
        """
        if not masses:
            return {self.theta: 1.0}
        if len(masses) == 1:
            return dict(masses[0])
        n = len(masses)
        discounted = []
        for i, m in enumerate(masses):
            k_bar = sum(self.conflict(m, masses[j]) for j in range(n) if j != i) / (n - 1)
            discounted.append(self.discount(m, 1.0 - k_bar))
        out = discounted[0]
        for m in discounted[1:]:
            out = self.combine(out, m)
        return out

    # ------------------------------------------------------------- bel / pl
    def belief(self, mass: Mass, hypothesis: frozenset[str]) -> float:
        """Bel(A) = sum of m(B) for B subset of A — committed support (lower bound)."""
        return sum(v for B, v in mass.items() if hypothesis >= B)

    def plausibility(self, mass: Mass, hypothesis: frozenset[str]) -> float:
        """Pl(A) = sum of m(B) for B intersecting A — not-ruled-out (upper bound)."""
        return sum(v for B, v in mass.items() if B & hypothesis)

    # --------------------------------------------------------------- decide
    def pignistic(self, mass: Mass) -> dict[str, float]:
        """BetP: split each focal set's mass uniformly over its members.

        Smets' pignistic transform — the credal->decision bridge. The uniform
        split is the insufficient-reason principle applied inside each focal set.
        """
        betp = {c: 0.0 for c in self.classes}
        for A, v in mass.items():
            share = v / len(A)
            for c in A:
                betp[c] += share
        return betp

    def decide(self, mass: Mass) -> tuple[str, float]:
        """Pignistic decision rule: argmax BetP.

        Returns (class_label, BetP confidence). Note BetP flattens the [Bel, Pl]
        interval — consumers needing "how uncertain" should read the interval
        via belief()/plausibility(), not infer it from this confidence.
        """
        betp = self.pignistic(mass)
        label = max(betp, key=betp.get)
        return label, betp[label]
