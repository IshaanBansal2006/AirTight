from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from airtight.contracts._hashing import content_hash
from airtight.contracts.site import XY

TacticFamily = Literal["charging_window", "decoy", "blind_spot", "comms_cut"]
TacticOrigin = Literal["hand", "random", "search", "llm"]


class Decoy(BaseModel):
    position: XY
    lead_time_s: Annotated[float, Field(ge=0)] = Field(
        description="seconds before the intruder enters that the decoy appears"
    )


class CommsEvent(BaseModel):
    t_s: Annotated[float, Field(ge=0)] = Field(description="seconds after the intruder enters")
    kind: Literal["cut_base_link", "jam_local"]


class Tactic(BaseModel):
    id: str
    family: TacticFamily
    entry_id: str
    phase: Annotated[float, Field(ge=0, lt=1)] = Field(
        description="fraction of the fleet's charge cycle at which the intruder enters"
    )
    speed_mps: Annotated[float, Field(gt=0)]
    waypoints: list[XY] = Field(
        min_length=1, description="path after the entry point; the last waypoint is the asset"
    )
    decoy: Decoy | None = None
    comms_event: CommsEvent | None = None
    origin: TacticOrigin = "hand"

    def content_hash(self) -> str:
        return content_hash(self)
