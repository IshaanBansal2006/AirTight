from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.redteam.config import LlmConfig

log = logging.getLogger(__name__)


class LlmCall(BaseModel):
    ts: datetime
    purpose: str
    model: str
    prompt_hash: str
    source: str
    input_tokens: int
    output_tokens: int
    usd: float


class BudgetExceededError(RuntimeError):
    pass


def cost_usd(cfg: LlmConfig, model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in cfg.usd_per_million_input or model not in cfg.usd_per_million_output:
        raise KeyError(
            f"no price for model {model!r}; add it to LlmConfig.usd_per_million_input/output before calling"
        )
    return (
        input_tokens * cfg.usd_per_million_input[model]
        + output_tokens * cfg.usd_per_million_output[model]
    ) / 1e6


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LlmClient:
    """JSON-schema completions with a content-addressed cache, an append-only ledger and a hard dollar cap.

    Resolution order per call: cache, then mock fixture if one is set, then the API. Tests only ever
    reach the first two.
    """

    def __init__(
        self, cfg: LlmConfig, cache_dir: Path, ledger_path: Path, mock_path: Path | None = None
    ) -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.ledger_path = ledger_path
        self.mock_path = mock_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def spent_usd(self) -> float:
        return sum(c.usd for c in self.ledger())

    def ledger(self) -> list[LlmCall]:
        if not self.ledger_path.exists():
            return []
        return [
            LlmCall.model_validate_json(line)
            for line in self.ledger_path.read_text().splitlines()
            if line.strip()
        ]

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str, purpose: str
    ) -> dict[str, Any]:
        key = hashlib.sha256(
            json.dumps([self.cfg.model, system, user, schema], sort_keys=True).encode()
        ).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            payload = json.loads(cached.read_text())
            self._record(purpose, key, "cache", 0, 0, 0.0)
            return dict(payload["content"])
        if self.mock_path is not None:
            content = json.loads(self.mock_path.read_text())
            self._record(purpose, key, "mock", 0, 0, 0.0)
            return dict(content)
        content, in_tok, out_tok = self._call_api(system, user, schema, schema_name)
        usd = cost_usd(self.cfg, self.cfg.model, in_tok, out_tok)
        cached.write_text(
            json.dumps(
                {
                    "model": self.cfg.model,
                    "content": content,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                }
            )
        )
        self._record(purpose, key, "api", in_tok, out_tok, usd)
        return content

    def _call_api(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str
    ) -> tuple[dict[str, Any], int, int]:
        worst_case = cost_usd(
            self.cfg, self.cfg.model, estimate_tokens(system + user), self.cfg.max_output_tokens
        )
        spent = self.spent_usd()
        if spent + worst_case > self.cfg.budget_usd:
            raise BudgetExceededError(
                f"spent ${spent:.4f} and this call could cost ${worst_case:.4f}, over the ${self.cfg.budget_usd:.2f} cap; "
                f"raise LlmConfig.budget_usd deliberately or use --mock / the cache"
            )
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            max_tokens=self.cfg.max_output_tokens,
            temperature=self.cfg.temperature,
        )
        choice = response.choices[0]
        if choice.message.content is None:
            raise RuntimeError(
                f"model returned no content (finish_reason={choice.finish_reason}); raise max_output_tokens or simplify the prompt"
            )
        usage = response.usage
        in_tok = int(usage.prompt_tokens) if usage else estimate_tokens(system + user)
        out_tok = int(usage.completion_tokens) if usage else estimate_tokens(choice.message.content)
        return json.loads(choice.message.content), in_tok, out_tok

    def _record(
        self, purpose: str, key: str, source: str, in_tok: int, out_tok: int, usd: float
    ) -> None:
        call = LlmCall(
            ts=datetime.now(UTC),
            purpose=purpose,
            model=self.cfg.model,
            prompt_hash=key[:12],
            source=source,
            input_tokens=in_tok,
            output_tokens=out_tok,
            usd=usd,
        )
        with self.ledger_path.open("a") as fh:
            fh.write(call.model_dump_json() + "\n")
        log.info(
            "llm %s via %s: %d in / %d out, $%.5f (total $%.4f)",
            purpose,
            source,
            in_tok,
            out_tok,
            usd,
            self.spent_usd(),
        )
