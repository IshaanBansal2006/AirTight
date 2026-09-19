from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from airtight.contracts import FleetConfig, SensorCurves, Site, TacticFamily
from airtight.redteam.config import RedTeamConfig
from airtight.redteam.families import FAMILIES, sample_tactic
from airtight.redteam.objective import EpisodeFn, evaluate

BAND = (0.6, 0.9)


class FamilyDifficulty(BaseModel):
    family: TacticFamily
    n_tactics: int
    mean_pd: float
    worst_pd: float


class DifficultyReport(BaseModel):
    n_seeds: int
    band: tuple[float, float]
    families: list[FamilyDifficulty]
    overall_mean_pd: float
    verdict: str

    def in_band(self) -> bool:
        return BAND[0] <= self.overall_mean_pd <= BAND[1]


def check_difficulty(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    cfg: RedTeamConfig | None = None,
    n_per_family: int = 30,
    families: Sequence[TacticFamily] = FAMILIES,
    master_seed: int = 0,
    workers: int = 1,
) -> DifficultyReport:
    """Random (unsearched) tactics against the baseline: is the scenario in the band where a fix can show?"""
    cfg = cfg or RedTeamConfig()
    rows: list[FamilyDifficulty] = []
    for i, fam in enumerate(families):
        rng = np.random.default_rng([master_seed, 99, i])
        tactics = [sample_tactic(fam, site, fleet, curves, rng, cfg) for _ in range(n_per_family)]
        scores = evaluate(
            tactics,
            site,
            fleet,
            curves,
            seeds,
            episode_fn,
            log_dir,
            cfg.search.margin_weight,
            workers,
        )
        pds = [1.0 - s.miss_rate for s in scores]
        rows.append(
            FamilyDifficulty(
                family=fam,
                n_tactics=n_per_family,
                mean_pd=float(np.mean(pds)),
                worst_pd=float(min(pds)),
            )
        )
    overall = float(np.mean([r.mean_pd for r in rows]))
    if overall > BAND[1]:
        verdict = f"too easy for the adversary: mean Pd {overall:.2f} > {BAND[1]}; widen entry points, cut sensor range or enlarge the site"
    elif overall < BAND[0]:
        verdict = f"too hard for the defender: mean Pd {overall:.2f} < {BAND[0]}; shorten paths, raise sensor range or add a fixed sensor"
    else:
        verdict = f"in band: mean Pd {overall:.2f} within [{BAND[0]}, {BAND[1]}]; freeze scenario parameters"
    return DifficultyReport(
        n_seeds=len(seeds), band=BAND, families=rows, overall_mean_pd=overall, verdict=verdict
    )
