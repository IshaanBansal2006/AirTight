from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
from airtight.redteam.config import RedTeamConfig
from airtight.redteam.families import FAMILIES, charge_cycle_s
from airtight.redteam.llm import LlmClient
from airtight.redteam.primitives import CompileError, Program, Step, compile_program
from airtight.redteam.search import SearchResult
from airtight.redteam.validate import validate

SCHEMA_NAME = "tactic_proposals"

_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "op": {
            "type": "string",
            "enum": ["enter", "wait", "move", "sprint", "drop_decoy", "cut_comms"],
        },
        "entry_id": {"type": ["string", "null"]},
        "x": {"type": ["number", "null"]},
        "y": {"type": ["number", "null"]},
        "seconds": {"type": ["number", "null"]},
        "lead_s": {"type": ["number", "null"]},
        "t_s": {"type": ["number", "null"]},
    },
    "required": ["op", "entry_id", "x", "y", "seconds", "lead_s", "t_s"],
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "family": {"type": "string", "enum": list(FAMILIES)},
                    "phase": {"type": "number"},
                    "speed_mps": {"type": "number"},
                    "rationale": {"type": "string"},
                    "steps": {"type": "array", "items": _STEP_SCHEMA},
                },
                "required": ["family", "phase", "speed_mps", "rationale", "steps"],
            },
        }
    },
    "required": ["proposals"],
}


class Rejected(BaseModel):
    index: int
    rationale: str
    errors: list[str]


class ProposalBatch(BaseModel):
    tactics: list[Tactic]
    rejected: list[Rejected] = Field(default_factory=list)
    rationales: dict[str, str] = Field(
        default_factory=dict, description="tactic id to the model's stated reasoning"
    )


def site_digest(site: Site, curves: SensorCurves) -> str:
    xmin, ymin, xmax, ymax = site.bounds
    lines = [
        f"Site '{site.name}', bounds x {xmin:.0f}..{xmax:.0f} m, y {ymin:.0f}..{ymax:.0f} m.",
        "Perimeter polygon: " + ", ".join(f"({p.x:.0f},{p.y:.0f})" for p in site.perimeter) + ".",
        f"Protected asset at ({site.asset.x:.0f},{site.asset.y:.0f}); responders need {site.response_time_s:.0f} s after an alarm.",
        "Entry points: "
        + "; ".join(
            f"{e.id} ({e.position.x:.0f},{e.position.y:.0f}, {e.kind})" for e in site.entry_points
        )
        + ".",
        "Docks: "
        + (
            "; ".join(f"{d.id} ({d.position.x:.0f},{d.position.y:.0f})" for d in site.docks)
            or "none"
        )
        + ".",
    ]
    if site.fixed_sensors:
        lines.append(
            "Fixed sensors: "
            + "; ".join(
                f"{s.id} at ({s.position.x:.0f},{s.position.y:.0f}) facing {s.heading_deg:.0f} deg, {curves.curves[s.sensor_type].fov_deg:.0f} deg FOV, {curves.curves[s.sensor_type].max_range_m():.0f} m range"
                for s in site.fixed_sensors
            )
            + "."
        )
    if site.benign_routes:
        lines.append(
            "Benign traffic: "
            + "; ".join(
                f"{r.cls} on route {r.id} at {r.arrival_rate_per_hour:.0f}/h"
                for r in site.benign_routes
            )
            + "."
        )
    return "\n".join(lines)


def patrol_summary(fleet: FleetConfig, curves: SensorCurves, cfg: RedTeamConfig) -> str:
    agents = "; ".join(
        f"{a.id} ({a.type}, {a.speed_mps:.1f} m/s, sensor {a.sensor_type} to {curves.curves[a.sensor_type].max_range_m():.0f} m, "
        f"endurance {a.endurance_s / 60:.0f} min, charge {a.charge_time_s / 60:.0f} min)"
        for a in fleet.agents
    )
    stagger = (
        "staggered"
        if fleet.charge_policy.stagger_offsets_s
        else "synchronized (all chargers start full together)"
    )
    return (
        f"Defender fleet '{fleet.name}': {agents}.\n"
        f"Charging is {stagger}; threshold {fleet.charge_policy.threshold_frac:.0%}; one charge cycle is {charge_cycle_s(fleet) / 60:.0f} min. "
        f"Phase is the fraction of that cycle at which the intruder enters.\n"
        f"Intruder speed must be between {cfg.speed_min_mps} and {cfg.speed_cap_mps} m/s; at most {cfg.max_waypoints} waypoints; "
        f"decoy lead time between {cfg.decoy_lead_s[0]:.0f} and {cfg.decoy_lead_s[1]:.0f} s."
    )


def results_table(prior: dict[str, SearchResult] | None) -> str:
    if not prior:
        return "No search results yet."
    rows = ["family | best adversary score | miss rate | entry | phase | speed | waypoints"]
    for fam, r in prior.items():
        t, s = r.best()
        rows.append(
            f"{fam} | {s.adversary_score:.2f} | {s.miss_rate:.2f} | {t.entry_id} | {t.phase:.2f} | {t.speed_mps:.1f} | {len(t.waypoints)}"
        )
    return "\n".join(rows)


def build_prompt(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    cfg: RedTeamConfig,
    prior: dict[str, SearchResult] | None,
    n: int,
) -> tuple[str, str]:
    system = (
        "You are a red-team planner probing a patrolled commercial site in simulation. "
        "Propose intrusion tactics as programs of primitives: enter(entry_id), wait(seconds) before entering to shift the phase, "
        "move(x,y), sprint(x,y) which moves at the speed cap, drop_decoy(x,y, lead_s) placed before entering, cut_comms(t_s) seconds after entry. "
        "Every program enters exactly once and ends at the asset. Exploit charging windows, decoys that pull the responder away, "
        "unwatched approaches, and link cuts. Make proposals diverse across families and entry points. Coordinates are metres inside the bounds."
    )
    user = (
        f"{site_digest(site, curves)}\n\n{patrol_summary(fleet, curves, cfg)}\n\n"
        f"Best tactics found so far by search:\n{results_table(prior)}\n\n"
        f"Propose {n} tactics that you expect to beat the table. Return JSON matching the schema."
    )
    return system, user


def propose(
    site: Site,
    fleet: FleetConfig,
    curves: SensorCurves,
    cfg: RedTeamConfig,
    client: LlmClient,
    prior: dict[str, SearchResult] | None = None,
    n: int | None = None,
    purpose: str = "propose",
) -> ProposalBatch:
    """One model call, schema-validated, compiled to tactics, validated against the site; invalid ones are kept as rejections."""
    n = n or cfg.llm.n_proposals
    system, user = build_prompt(site, fleet, curves, cfg, prior, n)
    raw = client.complete_json(system, user, RESPONSE_SCHEMA, SCHEMA_NAME, purpose)
    batch = ProposalBatch(tactics=[])
    for i, item in enumerate(raw.get("proposals", [])):
        try:
            prog = Program(
                family=item["family"],
                phase=item["phase"] % 1.0,
                speed_mps=item["speed_mps"],
                rationale=item.get("rationale", ""),
                steps=[Step(**s) for s in item["steps"]],
            )
            tactic = compile_program(prog, site, fleet, cfg)
        except (ValidationError, CompileError, KeyError, TypeError) as e:
            batch.rejected.append(
                Rejected(index=i, rationale=str(item.get("rationale", "")), errors=[str(e)])
            )
            continue
        errors = validate(tactic, site, cfg)
        if errors:
            batch.rejected.append(Rejected(index=i, rationale=prog.rationale, errors=errors))
            continue
        batch.tactics.append(tactic)
        batch.rationales[tactic.id] = prog.rationale
    return batch
