from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from airtight.contracts import (
    AgentSpec,
    ChargePolicy,
    FleetConfig,
    SensorCurves,
    Site,
    Tactic,
    TacticFamily,
)
from airtight.redteam.config import RedTeamConfig
from airtight.redteam.families import FAMILIES
from airtight.redteam.objective import EpisodeFn, evaluate
from airtight.redteam.search import search_all

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

log = logging.getLogger(__name__)

Move = str


class Candidate(BaseModel):
    move: Move
    fleet: FleetConfig
    worst_pd: float
    mean_pd: float
    cost_per_hour: float


class Iteration(BaseModel):
    k: int
    fleet_name: str
    cost_per_hour: float
    worst_pd_before_fix: float = Field(description="against the adversary that attacked this fleet")
    worst_pd_schedule_blind: float | None = Field(
        default=None,
        description="the same tactics with random entry phases: the adversary knows the site, not the schedule",
    )
    worst_tactic_id: str
    worst_family: TacticFamily
    move: Move | None = Field(
        default=None, description="the fix chosen at this iteration; None on the last"
    )
    candidates: list[Candidate] = Field(default_factory=list)


class MinMaxResult(BaseModel):
    site_hash: str
    seeds: list[int]
    budget_per_hour: float
    iterations: list[Iteration]


def _next_id(prefix: str, agents: Sequence[AgentSpec]) -> str:
    n = sum(1 for a in agents if a.id.startswith(prefix)) + 1
    return f"{prefix}_{n}"


def _template(agents: Sequence[AgentSpec], kind: str, fallback: AgentSpec) -> AgentSpec:
    return next((a for a in agents if a.type == kind), fallback)


DRONE = AgentSpec(
    id="drone_x",
    type="drone",
    speed_mps=8.0,
    endurance_s=1500.0,
    charge_time_s=2400.0,
    sensor_type="drone_camera",
)
GO2 = AgentSpec(
    id="go2_x",
    type="go2",
    speed_mps=1.2,
    endurance_s=5400.0,
    charge_time_s=3600.0,
    sensor_type="go2_camera",
)
GUARD = AgentSpec(
    id="guard_x",
    type="guard",
    speed_mps=1.4,
    endurance_s=28800.0,
    charge_time_s=0.0,
    sensor_type="human_eye",
)
COSTS = {"drone": 7.0, "go2": 9.0, "guard": 32.0}


def canonical_name(fleet: FleetConfig) -> str:
    n_drones = sum(a.type == "drone" for a in fleet.agents)
    parts = [
        f"d{n_drones}",
        "go2" if any(a.type == "go2" for a in fleet.agents) else "nogo2",
        "guard" if any(a.type == "guard" for a in fleet.agents) else "noguard",
        "stagger" if fleet.charge_policy.stagger_offsets_s else "sync",
    ]
    return "_".join(parts)


def stagger(fleet: FleetConfig) -> FleetConfig:
    drones = [a for a in fleet.agents if a.type == "drone"]
    if len(drones) < 2:
        return fleet
    cycle = drones[0].endurance_s + drones[0].charge_time_s
    offsets = {d.id: round(i * cycle / len(drones), 1) for i, d in enumerate(drones)}
    out = fleet.model_copy(
        update={
            "charge_policy": ChargePolicy(
                threshold_frac=fleet.charge_policy.threshold_frac, stagger_offsets_s=offsets
            ),
        }
    )
    return out.model_copy(update={"name": canonical_name(out)})


def add_agent(fleet: FleetConfig, kind: str) -> FleetConfig:
    template = _template(fleet.agents, kind, {"drone": DRONE, "go2": GO2, "guard": GUARD}[kind])
    new = template.model_copy(update={"id": _next_id(kind, fleet.agents)})
    agents = [*fleet.agents, new]
    costs = dict(fleet.cost_per_hour_by_type)
    costs.setdefault(kind, COSTS[kind])  # type: ignore[arg-type]
    out = fleet.model_copy(update={"agents": agents, "cost_per_hour_by_type": costs})
    out = out.model_copy(update={"name": canonical_name(out)})
    return stagger(out) if kind == "drone" and fleet.charge_policy.stagger_offsets_s else out


def moves(fleet: FleetConfig) -> dict[Move, FleetConfig]:
    out: dict[Move, FleetConfig] = {}
    if (
        not fleet.charge_policy.stagger_offsets_s
        and sum(a.type == "drone" for a in fleet.agents) >= 2
    ):
        out["stagger"] = stagger(fleet)
    out["add_drone"] = add_agent(fleet, "drone")
    if not any(a.type == "go2" for a in fleet.agents):
        out["add_go2"] = add_agent(fleet, "go2")
    if not any(a.type == "guard" for a in fleet.agents):
        out["add_guard"] = add_agent(fleet, "guard")
    return out


def choose(candidates: Sequence[Candidate], tolerance: float) -> Candidate:
    """Best worst case wins; worst cases within one seed's resolution tie, and ties go to the higher mean, then the lower cost."""
    top = max(c.worst_pd for c in candidates)
    tied = [c for c in candidates if c.worst_pd >= top - tolerance - 1e-9]
    return max(tied, key=lambda c: (c.mean_pd, -c.cost_per_hour))


def worst_case(
    fleet: FleetConfig,
    tactics: Sequence[Tactic],
    site: Site,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    workers: int,
    margin_weight: float,
    randomize_phase: bool = False,
) -> tuple[float, float, str]:
    scores = evaluate(
        tactics,
        site,
        fleet,
        curves,
        seeds,
        episode_fn,
        log_dir,
        margin_weight,
        workers,
        randomize_phase=randomize_phase,
    )
    pds = [(1.0 - s.miss_rate, s.tactic_id) for s in scores]
    worst_pd, worst_id = min(pds)
    return worst_pd, float(np.mean([p for p, _ in pds])), worst_id


def run_minmax(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    seeds: Sequence[int],
    episode_fn: EpisodeFn,
    log_dir: Path,
    out_dir: Path,
    iterations: int = 3,
    budget_per_hour: float = 80.0,
    per_family: int = 2,
    cfg: RedTeamConfig | None = None,
    workers: int = 1,
    master_seed: int = 0,
) -> MinMaxResult:
    """Attack, fix, re-attack: at each iteration the adversary searches the current fleet, every affordable
    defender move is scored against the tactics it found, the best worst-case wins, and the loop repeats."""
    cfg = cfg or RedTeamConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = MinMaxResult(
        site_hash=site.content_hash(),
        seeds=list(seeds),
        budget_per_hour=budget_per_hour,
        iterations=[],
    )
    current = fleet
    for k in range(iterations + 1):
        found = search_all(
            site,
            current,
            curves,
            seeds,
            episode_fn,
            log_dir,
            out_dir / f"iter{k}_tactics",
            FAMILIES,
            cfg,
            master_seed + k,
            workers,
            replay_seeds=1,
        )
        tactics = [t for fam in FAMILIES for t in found[fam].tactics[:per_family]]
        worst_pd, _, worst_id = worst_case(
            current,
            tactics,
            site,
            curves,
            seeds,
            episode_fn,
            log_dir,
            workers,
            cfg.search.margin_weight,
        )
        worst_family = next(
            fam for fam in FAMILIES if any(t.id == worst_id for t in found[fam].tactics)
        )
        blind_pd, _, _ = worst_case(
            current,
            tactics,
            site,
            curves,
            seeds,
            episode_fn,
            log_dir,
            workers,
            cfg.search.margin_weight,
            randomize_phase=True,
        )
        it = Iteration(
            k=k,
            fleet_name=current.name,
            cost_per_hour=current.cost_per_hour(),
            worst_pd_before_fix=worst_pd,
            worst_pd_schedule_blind=blind_pd,
            worst_tactic_id=worst_id,
            worst_family=worst_family,
        )
        log.info(
            "iteration %d: %s worst Pd %.2f (%s) at $%.0f/h",
            k,
            current.name,
            worst_pd,
            worst_id,
            current.cost_per_hour(),
        )
        if k == iterations:
            result.iterations.append(it)
            break
        for move, candidate in moves(current).items():
            if candidate.cost_per_hour() > budget_per_hour:
                continue
            c_worst, c_mean, _ = worst_case(
                candidate,
                tactics,
                site,
                curves,
                seeds,
                episode_fn,
                log_dir,
                workers,
                cfg.search.margin_weight,
            )
            it.candidates.append(
                Candidate(
                    move=move,
                    fleet=candidate,
                    worst_pd=c_worst,
                    mean_pd=c_mean,
                    cost_per_hour=candidate.cost_per_hour(),
                )
            )
        if not it.candidates:
            result.iterations.append(it)
            break
        best = choose(it.candidates, tolerance=1.0 / max(1, len(seeds)))
        it.move = best.move
        result.iterations.append(it)
        (out_dir / f"iter{k + 1}_fleet.json").write_text(best.fleet.model_dump_json(indent=2))
        current = best.fleet
        (out_dir / "minmax.json").write_text(result.model_dump_json(indent=2))
    (out_dir / "minmax.json").write_text(result.model_dump_json(indent=2))
    (out_dir / "summary.json").write_text(
        json.dumps(
            [
                {
                    "k": i.k,
                    "fleet": i.fleet_name,
                    "cost": i.cost_per_hour,
                    "worst_pd": i.worst_pd_before_fix,
                    "worst_pd_schedule_blind": i.worst_pd_schedule_blind,
                    "worst_family": i.worst_family,
                    "move": i.move,
                }
                for i in result.iterations
            ],
            indent=2,
        )
    )
    return result
