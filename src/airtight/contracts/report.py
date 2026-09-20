from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

Prob = Annotated[float, Field(ge=0, le=1)]
Interval = tuple[float, float]


class RocPoint(BaseModel):
    threshold: float
    pd: Prob
    pd_ci: Interval
    far_per_hour: Annotated[float, Field(ge=0)]


class PairedDelta(BaseModel):
    metric: str
    delta: float
    ci: Interval


class Conditions(BaseModel):
    """Every number on a slide carries these; the score line is meaningless without them."""

    far_per_hour_operating_point: float
    adversary_knowledge: str
    sensor_calibration: str
    detection_model_note: str
    seed_list_hash: str
    n_seeds: Annotated[int, Field(ge=1)]


class ConfigResult(BaseModel):
    config_name: str
    fleet_hash: str
    n_episodes: Annotated[int, Field(ge=1)]
    roc: list[RocPoint]
    pd_at_operating_point: Prob
    pd_at_operating_point_ci: Interval
    worst_tactic_id: str
    worst_tactic_pd: Prob
    worst_tactic_pd_schedule_blind: Prob | None = Field(
        default=None,
        description="the same worst tactics with their entry phase drawn at random per seed: an adversary that knows the site but not the charge schedule",
    )
    cost_per_hour: Annotated[float, Field(ge=0)]
    coverage_gap_s_per_hour: Annotated[float, Field(ge=0)]
    human_decisions_per_hour: Annotated[float, Field(ge=0)]
    paired_vs_baseline: list[PairedDelta] = Field(default_factory=list)


class Report(BaseModel):
    generated_at: datetime
    site_hash: str
    baseline_config: str
    conditions: Conditions
    configs: list[ConfigResult] = Field(min_length=1)

    def config(self, name: str) -> ConfigResult:
        for c in self.configs:
            if c.config_name == name:
                return c
        raise KeyError(
            f"no config {name!r} in report; known: {[c.config_name for c in self.configs]}"
        )
