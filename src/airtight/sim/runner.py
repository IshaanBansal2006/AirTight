from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from airtight.contracts import (
    EpisodeHeader,
    EpisodeResult,
    FleetConfig,
    OutcomeEvent,
    SensorCurves,
    Site,
    Tactic,
    write_episode_log,
)

if TYPE_CHECKING:
    from pathlib import Path

SIM_VERSION = "stub-0"


def run_episode(
    site: Site,
    fleet: FleetConfig,
    tactic: Tactic,
    sensor_curves: SensorCurves,
    seed: int,
    log_dir: Path,
) -> EpisodeResult:
    """Hour-1 stub: a random outcome with the real shape and a valid log. Lane B replaces this at H8."""
    rng = np.random.default_rng(seed)
    path_len = _path_length_m(site, tactic)
    t_reach_asset = path_len / tactic.speed_mps
    t_cdp = max(0.0, t_reach_asset - site.response_time_s)
    timely = bool(rng.random() < 0.7)
    t_alarm = (
        float(rng.uniform(0.0, t_cdp))
        if timely and t_cdp > 0
        else (float(rng.uniform(t_cdp, t_reach_asset)) if not timely else 0.0)
    )
    header = EpisodeHeader(
        site_hash=site.content_hash(),
        fleet_hash=fleet.content_hash(),
        sensor_curve_hash=sensor_curves.content_hash(),
        tactic=tactic,
        seed=seed,
        sim_version=SIM_VERSION,
    )
    outcome = OutcomeEvent(
        t=t_reach_asset,
        timely_detected=timely,
        t_alarm=t_alarm,
        t_cdp=t_cdp,
        human_decisions=int(timely),
    )
    log_path = log_dir / f"{fleet.name}__{tactic.id}__{seed}.jsonl"
    write_episode_log(log_path, header, [outcome])
    return EpisodeResult(timely_detected=timely, t_alarm=t_alarm, t_cdp=t_cdp, log_path=log_path)


def _path_length_m(site: Site, tactic: Tactic) -> float:
    pts = [site.entry(tactic.entry_id).position, *tactic.waypoints]
    xy = np.array([[p.x, p.y] for p in pts])
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
