from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

Pos = Annotated[float, Field(gt=0)]


class SearchConfig(BaseModel):
    n_random: Annotated[int, Field(ge=1)] = 200
    n_elite: Annotated[int, Field(ge=1)] = 10
    n_rounds: Annotated[int, Field(ge=0)] = 3
    n_children: Annotated[int, Field(ge=1)] = 4
    n_seeds: Annotated[int, Field(ge=1)] = 20
    margin_weight: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.1,
        description="weight of the continuous near-miss term that breaks ties between equal miss rates",
    )


class LlmConfig(BaseModel):
    model: str = "gpt-4.1-nano"
    budget_usd: Annotated[float, Field(ge=0)] = 0.50
    n_proposals: Annotated[int, Field(ge=1, le=12)] = 6
    max_output_tokens: Annotated[int, Field(ge=100)] = 1200
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.8
    usd_per_million_input: dict[str, float] = Field(
        default={
            "gpt-4.1-nano": 0.10,
            "gpt-4.1-mini": 0.40,
            "gpt-4o-mini": 0.15,
            "gpt-5-nano": 0.05,
        },
        description="price table used for accounting; verify against the provider's pricing page before quoting a number",
    )
    usd_per_million_output: dict[str, float] = Field(
        default={
            "gpt-4.1-nano": 0.40,
            "gpt-4.1-mini": 1.60,
            "gpt-4o-mini": 0.60,
            "gpt-5-nano": 0.40,
        }
    )


class RedTeamConfig(BaseModel):
    """Scenario parameters for the adversary. Frozen at the hour-10 sync; changes need a decision doc."""

    speed_min_mps: Pos = 0.8
    speed_cap_mps: Pos = 2.5
    max_waypoints: Annotated[int, Field(ge=1)] = 6
    perimeter_tol_m: Pos = 1.5
    asset_tol_m: Pos = 1.0
    min_leg_m: Pos = 2.0
    waypoint_jitter_m: Pos = 3.0
    coverage_cell_m: Pos = 5.0
    dock_halo_m: Pos = 15.0
    decoy_lead_s: tuple[float, float] = (10.0, 120.0)
    decoy_offset_m: Pos = 8.0
    search: SearchConfig = Field(default_factory=SearchConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
