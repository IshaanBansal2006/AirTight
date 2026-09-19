from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from airtight.contracts._hashing import content_hash

NonNeg = Annotated[float, Field(ge=0)]
Rate = Annotated[float, Field(ge=0, description="events per simulated hour")]


class XY(BaseModel, frozen=True):
    x: float
    y: float


class EntryPoint(BaseModel):
    id: str
    position: XY
    kind: str = "gate"


class Dock(BaseModel):
    id: str
    position: XY
    capacity: Annotated[int, Field(ge=1)] = 1


class FixedSensor(BaseModel):
    id: str
    position: XY
    sensor_type: str
    heading_deg: float = 0.0


class BenignRoute(BaseModel):
    id: str
    cls: str = Field(
        description="benign object class, matched against SensorCurve.pfa_per_look_by_class"
    )
    waypoints: list[XY] = Field(min_length=2)
    arrival_rate_per_hour: Rate


class Site(BaseModel):
    name: str
    bounds: tuple[float, float, float, float] = Field(
        description="xmin, ymin, xmax, ymax in metres"
    )
    perimeter: list[XY] = Field(min_length=3)
    entry_points: list[EntryPoint] = Field(min_length=1)
    asset: XY
    response_time_s: NonNeg = Field(
        description="time the responder needs after an alarm; defines the critical detection point"
    )
    docks: list[Dock] = Field(default_factory=list)
    fixed_sensors: list[FixedSensor] = Field(default_factory=list)
    benign_routes: list[BenignRoute] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> Site:
        xmin, ymin, xmax, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                f"bounds must be (xmin, ymin, xmax, ymax) with min < max, got {self.bounds}"
            )
        return self

    @model_validator(mode="after")
    def _unique_ids(self) -> Site:
        for label, items in (
            ("entry_points", self.entry_points),
            ("docks", self.docks),
            ("fixed_sensors", self.fixed_sensors),
            ("benign_routes", self.benign_routes),
        ):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} ids must be unique, got {ids}")
        return self

    def entry(self, entry_id: str) -> EntryPoint:
        for ep in self.entry_points:
            if ep.id == entry_id:
                return ep
        raise KeyError(
            f"no entry point {entry_id!r} in site {self.name!r}; known: {[e.id for e in self.entry_points]}"
        )

    def content_hash(self) -> str:
        return content_hash(self)
