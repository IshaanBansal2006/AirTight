from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from airtight.contracts.site import XY
from airtight.contracts.tactic import Tactic

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

Time = Annotated[float, Field(ge=0, description="simulated seconds since the intruder entered")]


class EpisodeHeader(BaseModel):
    site_hash: str
    fleet_hash: str
    sensor_curve_hash: str
    tactic: Tactic
    seed: int
    sim_version: str


class PositionEvent(BaseModel):
    kind: Literal["position"] = "position"
    t: Time
    object_id: str
    position: XY


class DetectionEvent(BaseModel):
    kind: Literal["detection"] = "detection"
    t: Time
    agent_id: str
    object_id: str
    true_positive: bool


class ScoreEvent(BaseModel):
    kind: Literal["score"] = "score"
    t: Time
    object_id: str
    value: float


class TaskEvent(BaseModel):
    kind: Literal["task"] = "task"
    t: Time
    task_id: str
    agent_id: str | None
    task_kind: str
    status: Literal["created", "assigned", "started", "done", "cancelled"]


class BatteryEvent(BaseModel):
    kind: Literal["battery"] = "battery"
    t: Time
    agent_id: str
    soc: Annotated[float, Field(ge=0, le=1)]
    docked: bool


class CommsGraphEvent(BaseModel):
    kind: Literal["comms_graph"] = "comms_graph"
    t: Time
    edges: list[tuple[str, str]]


class AlarmEvent(BaseModel):
    kind: Literal["alarm_delivered"] = "alarm_delivered"
    t: Time
    object_id: str
    via: str = Field(description="agent or link that carried the alarm to the human")


class OutcomeEvent(BaseModel):
    kind: Literal["outcome"] = "outcome"
    t: Time
    timely_detected: bool
    t_alarm: Time | None
    t_cdp: Time = Field(description="critical detection point: t_reach_asset minus response_time_s")
    human_decisions: Annotated[int, Field(ge=0)]


Event = Annotated[
    PositionEvent
    | DetectionEvent
    | ScoreEvent
    | TaskEvent
    | BatteryEvent
    | CommsGraphEvent
    | AlarmEvent
    | OutcomeEvent,
    Field(discriminator="kind"),
]
_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


class EpisodeResult(BaseModel):
    timely_detected: bool
    t_alarm: float | None
    t_cdp: float
    log_path: Path


def write_episode_log(path: Path, header: EpisodeHeader, events: Iterable[BaseModel]) -> None:
    """First line is the header, every following line one event. Append-friendly JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(header.model_dump_json() + "\n")
        for event in events:
            fh.write(event.model_dump_json() + "\n")


def read_episode_log(path: Path) -> tuple[EpisodeHeader, Iterator[BaseModel]]:
    """Header eagerly, events lazily so replay can stream a long log."""
    fh = path.open()
    first = fh.readline()
    if not first:
        fh.close()
        raise ValueError(f"{path} is empty; expected a header line then events")
    header = EpisodeHeader.model_validate_json(first)

    def events() -> Iterator[BaseModel]:
        with fh:
            for line in fh:
                if line.strip():
                    yield _event_adapter.validate_json(line)

    return header, events()
