from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from airtight.contracts import OutcomeEvent, ScoreEvent, read_episode_log

if TYPE_CHECKING:
    from pathlib import Path

INTRUDER_ID = "intruder"
DECOY_PREFIX = "decoy"


class EpisodeSummary(BaseModel):
    """What the scorer needs from one episode; peaks are threshold-free so any threshold can be applied later."""

    fleet_hash: str
    tactic_id: str
    family: str
    seed: int
    t_end_s: float
    t_cdp: float
    timely_at_ref: bool
    intruder_peak_before_cdp: float
    benign_peaks: dict[str, float] = Field(default_factory=dict)
    decoy_peak: float | None = None
    has_score_series: bool


def summarize_log(path: Path) -> EpisodeSummary:
    """Peaks from the score series; a header-plus-outcome log falls back to the outcome's verdict."""
    header, events = read_episode_log(path)
    intruder_peak = -math.inf
    benign: dict[str, float] = {}
    decoy: float | None = None
    outcome: OutcomeEvent | None = None
    has_scores = False
    t_cdp = 0.0
    for ev in events:
        if isinstance(ev, OutcomeEvent):
            outcome = ev
            t_cdp = ev.t_cdp
    if outcome is None:
        raise ValueError(f"{path} has no outcome event; the episode did not finish")
    _, events = read_episode_log(path)
    for ev in events:
        if not isinstance(ev, ScoreEvent):
            continue
        has_scores = True
        if ev.object_id == INTRUDER_ID:
            if ev.t <= t_cdp + 1e-9:
                intruder_peak = max(intruder_peak, ev.value)
        elif ev.object_id.startswith(DECOY_PREFIX):
            decoy = ev.value if decoy is None else max(decoy, ev.value)
        else:
            benign[ev.object_id] = max(benign.get(ev.object_id, -math.inf), ev.value)
    if not has_scores:
        intruder_peak = math.inf if outcome.timely_detected else -math.inf
    return EpisodeSummary(
        fleet_hash=header.fleet_hash,
        tactic_id=header.tactic.id,
        family=header.tactic.family,
        seed=header.seed,
        t_end_s=outcome.t,
        t_cdp=t_cdp,
        timely_at_ref=outcome.timely_detected,
        intruder_peak_before_cdp=intruder_peak,
        benign_peaks=benign,
        decoy_peak=decoy,
        has_score_series=has_scores,
    )


def summarize_dir(directory: Path) -> list[EpisodeSummary]:
    return [summarize_log(p) for p in sorted(directory.glob("*.jsonl"))]
