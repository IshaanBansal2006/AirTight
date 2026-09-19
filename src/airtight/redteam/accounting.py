from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel

from airtight.redteam.llm import LlmCall, LlmClient, cost_usd

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.redteam.config import LlmConfig


class LedgerSummary(BaseModel):
    calls: int
    api_calls: int
    cached_calls: int
    mock_calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    by_purpose: dict[str, float]
    mean_api_call_usd: float | None


def summarize_ledger(calls: list[LlmCall]) -> LedgerSummary:
    api = [c for c in calls if c.source == "api"]
    by_purpose: dict[str, float] = defaultdict(float)
    for c in calls:
        by_purpose[c.purpose] += c.usd
    return LedgerSummary(
        calls=len(calls),
        api_calls=len(api),
        cached_calls=sum(c.source == "cache" for c in calls),
        mock_calls=sum(c.source == "mock" for c in calls),
        input_tokens=sum(c.input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        usd=sum(c.usd for c in calls),
        by_purpose=dict(by_purpose),
        mean_api_call_usd=(sum(c.usd for c in api) / len(api)) if api else None,
    )


class PlannerCostComparison(BaseModel):
    """Cost of an LLM that plans every episode versus one that proposes once per scored configuration."""

    usd_per_call: float
    basis: str
    episodes_per_config: int
    n_configs: int
    calls_per_config_propose_then_search: int
    naive_usd: float
    propose_then_search_usd: float
    ratio: float


def compare_planners(
    cfg: LlmConfig,
    ledger: list[LlmCall],
    episodes_per_config: int,
    n_configs: int,
    calls_per_config: int = 2,
    prompt_tokens: int = 1400,
    output_tokens: int = 900,
) -> PlannerCostComparison:
    """Uses the measured mean cost of real calls when the ledger has any, else the price table on a typical prompt size."""
    summary = summarize_ledger(ledger)
    if summary.mean_api_call_usd is not None:
        per_call, basis = summary.mean_api_call_usd, f"mean of {summary.api_calls} real calls"
    else:
        per_call, basis = (
            cost_usd(cfg, cfg.model, prompt_tokens, output_tokens),
            f"price table at {prompt_tokens} in / {output_tokens} out tokens",
        )
    naive = per_call * episodes_per_config * n_configs
    pts = per_call * calls_per_config * n_configs
    return PlannerCostComparison(
        usd_per_call=per_call,
        basis=basis,
        episodes_per_config=episodes_per_config,
        n_configs=n_configs,
        calls_per_config_propose_then_search=calls_per_config,
        naive_usd=naive,
        propose_then_search_usd=pts,
        ratio=naive / pts if pts > 0 else float("inf"),
    )


def ledger_from(path: Path, cfg: LlmConfig) -> list[LlmCall]:
    return LlmClient(cfg, path.parent / "llm_cache", path).ledger()
