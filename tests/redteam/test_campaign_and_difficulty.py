from __future__ import annotations

from importlib import resources
from pathlib import Path

from airtight.contracts import FleetConfig, SensorCurves, Site
from airtight.redteam import RedTeamConfig
from airtight.redteam.campaign import run_campaign
from airtight.redteam.difficulty import BAND, check_difficulty
from airtight.redteam.llm import LlmClient
from airtight.sim.runner import run_episode

MOCK = Path(str(resources.files("airtight.redteam.fixtures").joinpath("proposals_mock.json")))


def test_difficulty_reports_band_and_verdict(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    rep = check_difficulty(
        site,
        fleet,
        curves,
        [1, 2, 3],
        run_episode,
        tmp_path,
        n_per_family=4,
        families=["charging_window", "decoy"],
    )
    assert rep.band == BAND and len(rep.families) == 2
    assert all(0.0 <= r.worst_pd <= r.mean_pd <= 1.0 for r in rep.families)
    assert rep.verdict and ("in band" in rep.verdict or "too" in rep.verdict)


def test_campaign_injects_llm_tactics_and_compares(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    cfg = RedTeamConfig()
    cfg.search.n_random, cfg.search.n_elite, cfg.search.n_rounds = 4, 3, 0
    client = LlmClient(cfg.llm, tmp_path / "cache", tmp_path / "ledger.jsonl", mock_path=MOCK)
    result, batch = run_campaign(
        site,
        fleet,
        curves,
        [1, 2],
        run_episode,
        tmp_path / "logs",
        tmp_path / "out",
        client,
        cfg,
        families=["charging_window", "decoy", "blind_spot"],
    )
    assert (
        result.proposals_accepted == 3
        and result.proposals_rejected == 1
        and result.llm_spent_usd == 0.0
    )
    assert {f.family for f in result.families} == {"charging_window", "decoy", "blind_spot"}
    assert result.any_llm_survivor()
    assert (tmp_path / "out" / "campaign.json").exists() and (
        tmp_path / "out" / "top_decoy.json"
    ).exists()
    assert all(f.with_llm_best >= 0.0 for f in result.families)
