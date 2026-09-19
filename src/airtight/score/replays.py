"""Export the baseline's worst failures as full logs for lane A to replay in dimOS."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from airtight.sim.constants import NEVER_SEEN
from airtight.sim.runner import ENGINE_ENV, run_episode

if TYPE_CHECKING:
    from pathlib import Path

    from airtight.score.report import ReportDetail
    from airtight.score.sweep import SweepResult

REPLAY_ENGINE = "v0"


def export_failures(
    result: SweepResult, detail: ReportDetail, n: int, out_dir: Path
) -> list[dict[str, Any]]:
    """For the baseline's worst tactic, rerun the n seeds the intruder was missed on by the
    largest margin below the operating threshold, with full logs, and write index.json.

    An intruder not seen at all before the critical detection point is the largest possible
    miss (it may still be seen later, too late to matter); ties keep seed order. The logs come from
    run_episode, the same call lanes A and C make. Its own verdict in each log is judged at
    TAU_REF, so the index also records the margin against this configuration's threshold.
    """
    inputs = result.inputs
    config = inputs.baseline
    tau = detail.configs[config].operating_threshold
    per_tactic = detail.configs[config].pd_by_tactic
    tactic_id = min(per_tactic, key=lambda t: per_tactic[t])  # first on ties, as in the report
    tactic = next(t for t in inputs.tactics if t.id == tactic_id)
    missed = [e for e in result.episodes[config][tactic_id] if e.intruder_peak < tau]
    missed.sort(key=lambda e: e.intruder_peak)  # stable: most negative first, seed order on ties

    out_dir.mkdir(parents=True, exist_ok=True)
    before = os.environ.get(ENGINE_ENV)
    os.environ[ENGINE_ENV] = REPLAY_ENGINE
    rows = []
    try:
        for episode in missed[:n]:
            outcome = run_episode(
                inputs.site,
                inputs.fleets[config],
                tactic,
                inputs.curves,
                episode.seed,
                out_dir,
                full_log=True,
            )
            never = episode.intruder_peak == NEVER_SEEN
            rows.append(
                {
                    "configuration": config,
                    "tactic_id": tactic_id,
                    "seed": episode.seed,
                    "outcome": (
                        "not seen before the critical detection point"
                        if never
                        else "seen before it, but scored below the operating threshold"
                    ),
                    "intruder_peak": None if never else episode.intruder_peak,
                    "operating_threshold": tau,
                    "timely_detected_at_tau_ref": outcome.timely_detected,
                    "t_alarm": outcome.t_alarm,
                    "t_cdp": outcome.t_cdp,
                    "log_path": str(outcome.log_path),
                }
            )
    finally:
        if before is None:
            os.environ.pop(ENGINE_ENV, None)
        else:
            os.environ[ENGINE_ENV] = before
    (out_dir / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows
