from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from airtight.redteam import RedTeamConfig, validate
from airtight.redteam.accounting import compare_planners, summarize_ledger
from airtight.redteam.llm import BudgetExceededError, LlmClient
from airtight.redteam.primitives import CompileError, Program, Step, compile_program
from airtight.redteam.proposer import RESPONSE_SCHEMA, build_prompt, propose

if TYPE_CHECKING:
    from airtight.contracts import FleetConfig, SensorCurves, Site

MOCK = Path(str(resources.files("airtight.redteam.fixtures").joinpath("proposals_mock.json")))


def _client(tmp_path: Path, mock: bool = True, budget: float = 0.5) -> LlmClient:
    cfg = RedTeamConfig().llm
    cfg.budget_usd = budget
    return LlmClient(
        cfg, tmp_path / "cache", tmp_path / "ledger.jsonl", mock_path=MOCK if mock else None
    )


def test_compile_program_shifts_phase_and_appends_asset(site: Site, fleet: FleetConfig) -> None:
    cfg = RedTeamConfig()
    prog = Program(
        family="charging_window",
        phase=0.1,
        speed_mps=1.5,
        steps=[
            Step(op="wait", seconds=600),
            Step(op="enter", entry_id="north_gate"),
            Step(op="sprint", x=60, y=60),
        ],
    )
    t = compile_program(prog, site, fleet, cfg)
    assert t.phase > 0.1 and t.speed_mps == cfg.speed_cap_mps
    assert t.waypoints[-1] == site.asset and t.origin == "llm"


def test_compile_program_rejects_programs_without_enter(site: Site, fleet: FleetConfig) -> None:
    with pytest.raises(CompileError, match="never enters"):
        compile_program(
            Program(family="decoy", phase=0.0, speed_mps=1.0, steps=[Step(op="move", x=1, y=1)]),
            site,
            fleet,
            RedTeamConfig(),
        )


def test_propose_with_mock_validates_and_rejects(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    client = _client(tmp_path)
    batch = propose(site, fleet, curves, RedTeamConfig(), client)
    assert len(batch.tactics) == 3 and len(batch.rejected) == 1
    assert all(validate(t, site) == [] and t.origin == "llm" for t in batch.tactics)
    assert any("speed" in e for e in batch.rejected[0].errors)
    assert {t.family for t in batch.tactics} == {"charging_window", "decoy", "blind_spot"}


def test_ledger_records_mock_and_cache_at_zero_cost(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    client = _client(tmp_path)
    propose(site, fleet, curves, RedTeamConfig(), client)
    propose(site, fleet, curves, RedTeamConfig(), client)
    calls = client.ledger()
    assert [c.source for c in calls] == ["mock", "mock"] and client.spent_usd() == 0.0
    summary = summarize_ledger(calls)
    assert summary.mock_calls == 2 and summary.api_calls == 0


def test_cache_hit_short_circuits_before_budget(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    cfg = RedTeamConfig()
    system, user = build_prompt(site, fleet, curves, cfg, None, cfg.llm.n_proposals)
    live = _client(tmp_path, mock=False, budget=0.0)
    import hashlib

    key = hashlib.sha256(
        json.dumps([live.cfg.model, system, user, RESPONSE_SCHEMA], sort_keys=True).encode()
    ).hexdigest()
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache" / f"{key}.json").write_text(
        json.dumps(
            {
                "model": live.cfg.model,
                "content": json.loads(MOCK.read_text()),
                "input_tokens": 1,
                "output_tokens": 1,
            }
        )
    )
    batch = propose(site, fleet, curves, cfg, live)
    assert len(batch.tactics) == 3 and live.ledger()[0].source == "cache"


def test_budget_cap_refuses_before_any_network(
    site: Site, fleet: FleetConfig, curves: SensorCurves, tmp_path: Path
) -> None:
    live = _client(tmp_path, mock=False, budget=0.0)
    with pytest.raises(BudgetExceededError, match="cap"):
        propose(site, fleet, curves, RedTeamConfig(), live)


def test_compare_planners_uses_price_table_without_real_calls() -> None:
    cmp = compare_planners(RedTeamConfig().llm, [], episodes_per_config=200, n_configs=12)
    assert cmp.naive_usd > cmp.propose_then_search_usd and "price table" in cmp.basis
    assert cmp.ratio == pytest.approx(200 / 2)
