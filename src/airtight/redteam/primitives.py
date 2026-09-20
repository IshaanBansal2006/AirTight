from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

from airtight.contracts import XY, CommsEvent, Decoy, FleetConfig, Site, Tactic, TacticFamily
from airtight.redteam.families import charge_cycle_s

if TYPE_CHECKING:
    from airtight.redteam.config import RedTeamConfig

Op = Literal["enter", "wait", "move", "sprint", "drop_decoy", "cut_comms"]


class Step(BaseModel):
    """One primitive in the adversary's vocabulary; unused fields are null."""

    op: Op
    entry_id: str | None = None
    x: float | None = None
    y: float | None = None
    seconds: float | None = None
    lead_s: float | None = None
    t_s: float | None = None


class Program(BaseModel):
    family: TacticFamily
    phase: Annotated[float, Field(ge=0, lt=1)]
    speed_mps: Annotated[float, Field(gt=0)]
    steps: list[Step] = Field(min_length=1)
    rationale: str = ""


class CompileError(ValueError):
    pass


def compile_program(prog: Program, site: Site, fleet: FleetConfig, cfg: RedTeamConfig) -> Tactic:
    """Turn a primitive program into a Tactic. Waits before `enter` shift the phase; `sprint` sets the cap speed."""
    phase = prog.phase
    speed = prog.speed_mps
    entry_id: str | None = None
    waypoints: list[XY] = []
    decoy: Decoy | None = None
    comms: CommsEvent | None = None
    cycle = charge_cycle_s(fleet)
    for i, s in enumerate(prog.steps):
        if s.op == "enter":
            if entry_id is not None:
                raise CompileError(f"step {i}: second enter; a program enters once")
            if not s.entry_id:
                raise CompileError(f"step {i}: enter needs entry_id")
            entry_id = s.entry_id
        elif s.op == "wait":
            if entry_id is not None:
                raise CompileError(
                    f"step {i}: wait after enter is not modelled; waits only shift the entry phase"
                )
            phase = (phase + (s.seconds or 0.0) / cycle) % 1.0
        elif s.op in ("move", "sprint"):
            if s.x is None or s.y is None:
                raise CompileError(f"step {i}: {s.op} needs x and y")
            waypoints.append(XY(x=s.x, y=s.y))
            if s.op == "sprint":
                speed = cfg.speed_cap_mps
        elif s.op == "drop_decoy":
            if s.x is None or s.y is None:
                raise CompileError(f"step {i}: drop_decoy needs x and y")
            decoy = Decoy(
                position=XY(x=s.x, y=s.y),
                lead_time_s=float(s.lead_s if s.lead_s is not None else cfg.decoy_lead_s[0]),
            )
        elif s.op == "cut_comms":
            comms = CommsEvent(t_s=float(s.t_s or 0.0), kind="cut_base_link")
    if entry_id is None:
        raise CompileError("program never enters the site")
    if not waypoints or _dist(waypoints[-1], site.asset) > cfg.asset_tol_m:
        waypoints.append(site.asset)
    digest = hashlib.sha256(prog.model_dump_json().encode()).hexdigest()[:8]
    return Tactic(
        id=f"llm-{prog.family}-{digest}",
        family=prog.family,
        entry_id=entry_id,
        phase=phase,
        speed_mps=speed,
        waypoints=waypoints[: cfg.max_waypoints],
        decoy=decoy,
        comms_event=comms,
        origin="llm",
    )


def _dist(a: XY, b: XY) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)
