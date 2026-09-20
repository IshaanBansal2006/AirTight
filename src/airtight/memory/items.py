from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class CoverageCell(BaseModel):
    """When a cell was last watched. Merges by max: later sightings never lose to earlier ones."""

    kind: Literal["coverage"] = "coverage"
    cx: int
    cy: int
    last_seen_t: float
    by: str


class Claim(BaseModel):
    """Which agent holds a task. Merges last-writer-wins on t, tie broken by agent id so order cannot matter."""

    kind: Literal["claim"] = "claim"
    task_id: str
    agent_id: str
    t: float


class Evidence(BaseModel):
    """One detection with its score. Merges as a set keyed by evidence_id (conflicting content resolved by a
    total order on the serialized item, so replicas agree); an object's score is the sum over its evidence."""

    kind: Literal["evidence"] = "evidence"
    evidence_id: str
    object_id: str
    agent_id: str
    t: float
    score: float
    x: float
    y: float


MemoryItem = Annotated[CoverageCell | Claim | Evidence, Field(discriminator="kind")]
item_adapter: TypeAdapter[MemoryItem] = TypeAdapter(MemoryItem)

Key = tuple[str, str]


def key_of(item: CoverageCell | Claim | Evidence) -> Key:
    if isinstance(item, CoverageCell):
        return ("coverage", f"{item.cx},{item.cy}")
    if isinstance(item, Claim):
        return ("claim", item.task_id)
    return ("evidence", item.evidence_id)
