from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from airtight.contracts._hashing import content_hash

AgentType = Literal["drone", "go2", "guard"]
CommsMode = Literal["central_perfect", "central_cut", "replicated_cut"]
Pos = Annotated[float, Field(gt=0)]
NonNeg = Annotated[float, Field(ge=0)]


class AgentSpec(BaseModel):
    id: str
    type: AgentType
    speed_mps: Pos
    endurance_s: Pos = Field(description="operating time from full charge to the charge threshold")
    charge_time_s: NonNeg = Field(description="dock time from threshold back to full; 0 for guards")
    sensor_type: str = Field(description="key into SensorCurves.curves")


class ChargePolicy(BaseModel):
    threshold_frac: Annotated[float, Field(gt=0, lt=1)] = 0.2
    stagger_offsets_s: dict[str, NonNeg] = Field(
        default_factory=dict,
        description="agent id to initial phase offset; empty means synchronized",
    )


class FleetConfig(BaseModel):
    name: str
    agents: list[AgentSpec] = Field(min_length=1)
    charge_policy: ChargePolicy = Field(default_factory=ChargePolicy)
    comms_mode: CommsMode = "central_perfect"
    cost_per_hour_by_type: dict[AgentType, NonNeg]

    @model_validator(mode="after")
    def _consistent(self) -> FleetConfig:
        ids = [a.id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"agent ids must be unique, got {ids}")
        unknown = set(self.charge_policy.stagger_offsets_s) - set(ids)
        if unknown:
            raise ValueError(f"stagger offsets reference unknown agents {sorted(unknown)}")
        missing = {a.type for a in self.agents} - set(self.cost_per_hour_by_type)
        if missing:
            raise ValueError(f"cost_per_hour_by_type lacks {sorted(missing)}")
        return self

    def cost_per_hour(self) -> float:
        return sum(self.cost_per_hour_by_type[a.type] for a in self.agents)

    def content_hash(self) -> str:
        return content_hash(self)
