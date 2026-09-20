from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from airtight.contracts import EpisodeResult, FleetConfig, SensorCurves, Site, Tactic
from airtight.redteam import RedTeamConfig
from airtight.redteam.objective import summarize
from airtight.redteam.search import search_all, search_family
from airtight.sim.runner import run_episode

EXAMPLES = resources.files("airtight.contracts.examples")


def test_summarize_orders_near_misses_below_misses(hand_tactics: list[Tactic]) -> None:
    t = hand_tactics[0]
    miss = EpisodeResult(timely_detected=False, t_alarm=None, t_cdp=100.0, log_path=Path("x"))
    close = EpisodeResult(timely_detected=True, t_alarm=95.0, t_cdp=100.0, log_path=Path("x"))
    early = EpisodeResult(timely_detected=True, t_alarm=5.0, t_cdp=100.0, log_path=Path("x"))
    s_miss, s_close, s_early = (summarize(t, [r], 90.0, 0.1) for r in (miss, close, early))
    assert s_miss.adversary_score > s_close.adversary_score > s_early.adversary_score
    assert s_miss.miss_rate == 1.0 and s_close.miss_rate == 0.0


def test_search_family_runs_and_ranks(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    cfg = RedTeamConfig()
    cfg.search.n_random, cfg.search.n_elite, cfg.search.n_rounds, cfg.search.n_children = 8, 3, 1, 2
    res = search_family(
        "charging_window",
        site,
        fleet,
        curves,
        [1, 2, 3],
        run_episode,
        tmp_path,
        cfg,
        np.random.default_rng(0),
    )
    scores = [s.adversary_score for s in res.scores]
    assert scores == sorted(scores, reverse=True) and len(res.tactics) == 3
    assert res.n_episodes == (8 + 3 * 2) * 3


def test_search_all_writes_outputs_and_is_deterministic(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    cfg = RedTeamConfig()
    cfg.search.n_random, cfg.search.n_elite, cfg.search.n_rounds = 6, 2, 0
    a = search_all(
        site,
        fleet,
        curves,
        [1, 2],
        run_episode,
        tmp_path / "logs",
        tmp_path / "a",
        families=["decoy", "comms_cut"],
        cfg=cfg,
    )
    b = search_all(
        site,
        fleet,
        curves,
        [1, 2],
        run_episode,
        tmp_path / "logs",
        tmp_path / "b",
        families=["decoy", "comms_cut"],
        cfg=cfg,
    )
    assert a["decoy"].tactics[0].id == b["decoy"].tactics[0].id
    summary = json.loads((tmp_path / "a" / "summary.json").read_text())
    assert set(summary["families"]) == {"decoy", "comms_cut"}
    assert (tmp_path / "a" / "top_decoy.json").exists()


def test_injected_tactics_join_population(
    site: Site, fleet: FleetConfig, curves: SensorCurves, hand_tactics: list[Tactic], tmp_path: Path
) -> None:
    cfg = RedTeamConfig()
    cfg.search.n_random, cfg.search.n_elite, cfg.search.n_rounds = 2, 3, 0
    res = search_family(
        "decoy",
        site,
        fleet,
        curves,
        [1],
        run_episode,
        tmp_path,
        cfg,
        np.random.default_rng(0),
        seed_tactics=hand_tactics,
    )
    assert any(t.id == "hand_decoy_loading_dock" for t in res.tactics)


def test_schedule_blind_changes_only_the_phase_and_is_seed_deterministic() -> None:
    from airtight.redteam.objective import schedule_blind

    tactic = Tactic.model_validate_json(EXAMPLES.joinpath("tactic.json").read_text())
    a, b, c = schedule_blind(tactic, 11), schedule_blind(tactic, 11), schedule_blind(tactic, 12)
    assert a == b and a.phase != c.phase
    assert a.model_dump(exclude={"phase"}) == tactic.model_dump(exclude={"phase"})
    assert 0.0 <= a.phase < 1.0
