from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic, TacticFamily
from airtight.redteam.config import RedTeamConfig
from airtight.redteam.coverage import CoverageMap, GeometryCoverage
from airtight.redteam.families import FAMILIES, perturb, sample_tactic
from airtight.redteam.objective import EpisodeFn, TacticScore, evaluate, write_replay_logs
from airtight.redteam.validate import validate

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

log = logging.getLogger(__name__)


class EvaluatedTactic(BaseModel):
    round: int
    tactic: Tactic
    score: TacticScore


class SearchResult(BaseModel):
    family: TacticFamily
    seeds: list[int]
    n_episodes: int
    scores: list[TacticScore] = Field(description="elites, best adversary score first")
    tactics: list[Tactic] = Field(description="same order as scores")
    evaluated: list[EvaluatedTactic] = Field(
        default_factory=list, description="every tactic scored, by round, elites included"
    )

    def best(self) -> tuple[Tactic, TacticScore]:
        return self.tactics[0], self.scores[0]


def load_seeds(path: Path, n: int) -> list[int]:
    seeds = json.loads(path.read_text())["seeds"]
    if n > len(seeds):
        raise ValueError(
            f"asked for {n} seeds but {path} holds {len(seeds)}; lower n_seeds or extend the committed list"
        )
    return [int(s) for s in seeds[:n]]


def search_family(
    family: TacticFamily,
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    cfg: RedTeamConfig | None = None,
    rng: np.random.Generator | None = None,
    workers: int = 1,
    seed_tactics: Sequence[Tactic] = (),
    coverage: CoverageMap | None = None,
) -> SearchResult:
    """Random sampling, keep the elites, refine each by perturbation for a fixed number of rounds."""
    cfg = cfg or RedTeamConfig()
    rng = rng or np.random.default_rng(0)
    sc = cfg.search
    if coverage is None and family == "blind_spot":
        coverage = GeometryCoverage(cfg.coverage_cell_m, cfg.dock_halo_m).coverage(site, curves)
    injected = [t for t in seed_tactics if t.family == family and not validate(t, site, cfg)]
    population = injected + [
        sample_tactic(family, site, fleet, curves, rng, cfg, coverage) for _ in range(sc.n_random)
    ]
    n_episodes = 0

    evaluated: list[tuple[int, Tactic, TacticScore]] = []
    round_no = 0

    def run(ts: list[Tactic]) -> list[TacticScore]:
        nonlocal n_episodes
        n_episodes += len(ts) * len(seeds)
        scores = evaluate(
            ts, site, fleet, curves, seeds, episode_fn, log_dir, sc.margin_weight, workers
        )
        evaluated.extend((round_no, t, s) for t, s in zip(ts, scores, strict=True))
        return scores

    ranked = _rank(population, run(population))
    elites = ranked[: sc.n_elite]
    log.info(
        "%s: round 0 best=%.3f (miss %.2f)",
        family,
        elites[0][1].adversary_score,
        elites[0][1].miss_rate,
    )
    for r in range(1, sc.n_rounds + 1):
        round_no = r
        children = [perturb(t, site, rng, cfg) for t, _ in elites for _ in range(sc.n_children)]
        children = _dedupe(children, {t.id for t, _ in elites})
        pool = elites + _rank(children, run(children))
        elites = sorted(pool, key=lambda p: p[1].adversary_score, reverse=True)[: sc.n_elite]
        log.info(
            "%s: round %d best=%.3f (miss %.2f)",
            family,
            r,
            elites[0][1].adversary_score,
            elites[0][1].miss_rate,
        )
    return SearchResult(
        family=family,
        seeds=list(seeds),
        n_episodes=n_episodes,
        scores=[s for _, s in elites],
        tactics=[t for t, _ in elites],
        evaluated=[EvaluatedTactic(round=r, tactic=t, score=s) for r, t, s in evaluated],
    )


def _rank(tactics: list[Tactic], scores: list[TacticScore]) -> list[tuple[Tactic, TacticScore]]:
    return sorted(
        zip(tactics, scores, strict=True), key=lambda p: p[1].adversary_score, reverse=True
    )


def _dedupe(tactics: list[Tactic], taken: set[str]) -> list[Tactic]:
    out: list[Tactic] = []
    for t in tactics:
        if t.id not in taken:
            taken.add(t.id)
            out.append(t)
    return out


def search_all(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    out_dir: Path,
    families: Sequence[TacticFamily] = FAMILIES,
    cfg: RedTeamConfig | None = None,
    master_seed: int = 0,
    workers: int = 1,
    seed_tactics: Sequence[Tactic] = (),
    replay_seeds: int = 3,
) -> dict[TacticFamily, SearchResult]:
    """One search per family with its own derived RNG; writes top_<family>.json, summary.json and replay logs."""
    cfg = cfg or RedTeamConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[TacticFamily, SearchResult] = {}
    for i, family in enumerate(families):
        rng = np.random.default_rng([master_seed, i])
        res = search_family(
            family, site, fleet, curves, seeds, episode_fn, log_dir, cfg, rng, workers, seed_tactics
        )
        (out_dir / f"top_{family}.json").write_text(res.model_dump_json(indent=2))
        results[family] = res
    if replay_seeds > 0:
        write_replay_logs(
            [r.tactics[0] for r in results.values()],
            site,
            fleet,
            curves,
            list(seeds)[:replay_seeds],
            episode_fn,
            out_dir / "replays",
        )
    summary = {
        fam: {
            "best_tactic_id": r.tactics[0].id,
            "best_origin": r.tactics[0].origin,
            "adversary_score": r.scores[0].adversary_score,
            "miss_rate": r.scores[0].miss_rate,
            "mean_margin_s": r.scores[0].mean_margin_s,
            "n_episodes": r.n_episodes,
        }
        for fam, r in results.items()
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"site": site.name, "fleet": fleet.name, "n_seeds": len(seeds), "families": summary},
            indent=2,
        )
    )
    return results
