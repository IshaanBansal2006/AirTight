from __future__ import annotations

import json
from typing import TYPE_CHECKING

from airtight.redteam import RedTeamConfig
from airtight.redteam.minmax import Candidate, add_agent, choose, moves, run_minmax, stagger
from airtight.sim.runner import run_episode

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.contracts import FleetConfig, SensorCurves, Site


def test_moves_add_agents_and_stagger(fleet: FleetConfig) -> None:
    m = moves(fleet)
    assert "stagger" in m and m["stagger"].charge_policy.stagger_offsets_s
    assert "add_drone" in m and sum(a.type == "drone" for a in m["add_drone"].agents) == 3
    assert "add_go2" not in m and "add_guard" not in m
    assert (
        add_agent(fleet, "drone").cost_per_hour()
        == fleet.cost_per_hour() + fleet.cost_per_hour_by_type["drone"]
    )
    assert (
        stagger(stagger(fleet)).charge_policy.stagger_offsets_s
        == stagger(fleet).charge_policy.stagger_offsets_s
    )


def test_minmax_runs_two_iterations_on_the_stub(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    cfg = RedTeamConfig()
    cfg.search.n_random, cfg.search.n_elite, cfg.search.n_rounds = 4, 2, 0
    res = run_minmax(
        site,
        fleet,
        curves,
        [1, 2],
        run_episode,
        tmp_path / "logs",
        tmp_path / "out",
        iterations=1,
        budget_per_hour=80.0,
        per_family=1,
        cfg=cfg,
    )
    assert (
        len(res.iterations) == 2
        and res.iterations[0].move is not None
        and res.iterations[1].move is None
    )
    assert (tmp_path / "out" / "minmax.json").exists() and (
        tmp_path / "out" / "iter1_fleet.json"
    ).exists()
    assert all(c.cost_per_hour <= 80.0 for c in res.iterations[0].candidates)
    assert all(0.0 <= (i.worst_pd_schedule_blind or 0.0) <= 1.0 for i in res.iterations)
    rows = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert all(r["worst_pd_schedule_blind"] is not None for r in rows)


def test_choose_breaks_one_seed_ties_on_the_mean(fleet: FleetConfig) -> None:
    a = Candidate(move="stagger", fleet=fleet, worst_pd=0.00, mean_pd=0.66, cost_per_hour=55.0)
    b = Candidate(move="add_drone", fleet=fleet, worst_pd=0.05, mean_pd=0.11, cost_per_hour=62.0)
    assert choose([a, b], tolerance=0.05).move == "stagger"
    c = Candidate(move="add_go2", fleet=fleet, worst_pd=0.30, mean_pd=0.40, cost_per_hour=64.0)
    assert choose([a, b, c], tolerance=0.05).move == "add_go2"
