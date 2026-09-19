from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from airtight.contracts import FleetConfig, SensorCurves, Site, TacticFamily
from airtight.redteam.config import RedTeamConfig
from airtight.redteam.families import FAMILIES
from airtight.redteam.llm import LlmClient
from airtight.redteam.objective import EpisodeFn
from airtight.redteam.proposer import ProposalBatch, propose
from airtight.redteam.search import SearchResult, search_all


class FamilyOutcome(BaseModel):
    family: TacticFamily
    search_only_best: float
    with_llm_best: float
    llm_tactic_in_elites: bool
    llm_tactic_is_best: bool


class CampaignResult(BaseModel):
    proposals_accepted: int
    proposals_rejected: int
    llm_spent_usd: float
    families: list[FamilyOutcome]
    n_episodes: int

    def any_llm_survivor(self) -> bool:
        return any(f.llm_tactic_in_elites for f in self.families)


def run_campaign(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    out_dir: Path,
    client: LlmClient,
    cfg: RedTeamConfig | None = None,
    families: Sequence[TacticFamily] = FAMILIES,
    master_seed: int = 0,
    workers: int = 1,
) -> tuple[CampaignResult, ProposalBatch]:
    """Search, propose against the search's table, search again with the proposals injected, compare.

    The same master seed drives both searches so the only difference is the injected tactics.
    """
    cfg = cfg or RedTeamConfig()
    base = search_all(
        site,
        fleet,
        curves,
        seeds,
        episode_fn,
        log_dir,
        out_dir / "search_only",
        families,
        cfg,
        master_seed,
        workers,
    )
    batch = propose(
        site,
        fleet,
        curves,
        cfg,
        client,
        prior={str(k): v for k, v in base.items()},
        purpose="campaign",
    )
    boosted = search_all(
        site,
        fleet,
        curves,
        seeds,
        episode_fn,
        log_dir,
        out_dir / "with_llm",
        families,
        cfg,
        master_seed,
        workers,
        seed_tactics=batch.tactics,
    )
    outcomes = [_outcome(fam, base[fam], boosted[fam]) for fam in families]
    result = CampaignResult(
        proposals_accepted=len(batch.tactics),
        proposals_rejected=len(batch.rejected),
        llm_spent_usd=client.spent_usd(),
        families=outcomes,
        n_episodes=sum(r.n_episodes for r in base.values())
        + sum(r.n_episodes for r in boosted.values()),
    )
    (out_dir / "campaign.json").write_text(result.model_dump_json(indent=2))
    (out_dir / "llm_proposals.json").write_text(batch.model_dump_json(indent=2))
    for fam in families:
        (out_dir / f"top_{fam}.json").write_text(boosted[fam].model_dump_json(indent=2))
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"campaign": True, "families": {f.family: f.model_dump() for f in outcomes}}, indent=2
        )
    )
    return result, batch


def _outcome(fam: TacticFamily, base: SearchResult, boosted: SearchResult) -> FamilyOutcome:
    return FamilyOutcome(
        family=fam,
        search_only_best=base.scores[0].adversary_score,
        with_llm_best=boosted.scores[0].adversary_score,
        llm_tactic_in_elites=any(t.origin == "llm" for t in boosted.tactics),
        llm_tactic_is_best=boosted.tactics[0].origin == "llm",
    )
