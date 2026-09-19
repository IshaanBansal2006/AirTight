from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from airtight.contracts._hashing import content_hash

Prob = Annotated[float, Field(ge=0, le=1)]


class SensorCurve(BaseModel):
    sensor_type: str
    range_bins_m: list[Annotated[float, Field(gt=0)]] = Field(
        min_length=1, description="upper edge of each range bin, ascending"
    )
    pd_per_look: list[Prob] = Field(
        description="detection probability per look for a target in each bin"
    )
    pfa_per_look_by_class: dict[str, Prob] = Field(
        description="false-positive probability per look per benign class"
    )
    fov_deg: Annotated[float, Field(gt=0, le=360)]
    look_rate_hz: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def _aligned(self) -> SensorCurve:
        if len(self.pd_per_look) != len(self.range_bins_m):
            raise ValueError(
                f"{self.sensor_type}: pd_per_look has {len(self.pd_per_look)} entries for {len(self.range_bins_m)} range bins"
            )
        if self.range_bins_m != sorted(self.range_bins_m):
            raise ValueError(
                f"{self.sensor_type}: range_bins_m must be ascending, got {self.range_bins_m}"
            )
        return self

    def max_range_m(self) -> float:
        return self.range_bins_m[-1]


class SensorCurves(BaseModel):
    source: str = Field(description="how the curve was produced, quoted on the score line")
    curves: dict[str, SensorCurve]

    @model_validator(mode="after")
    def _keys_match(self) -> SensorCurves:
        bad = [k for k, c in self.curves.items() if c.sensor_type != k]
        if bad:
            raise ValueError(f"curve keys must equal their sensor_type, mismatched: {bad}")
        return self

    def content_hash(self) -> str:
        return content_hash(self)
