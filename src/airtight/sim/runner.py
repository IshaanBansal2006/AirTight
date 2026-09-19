from __future__ import annotations

import os
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
from airtight.sim.episode import EpisodeParams, simulate
from airtight.sim.recorder import LogRecorder, outcome_event

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

SIM_VERSION = "stub-0"
SIM_VERSION_V0 = "v0"
ENGINE_ENV = "AIRTIGHT_ENGINE"
ENGINES = ("stub", "v0")


def run_episode(
    site: Site,
    fleet: FleetConfig,
    tactic: Tactic,
    sensor_curves: SensorCurves,
    seed: int,
    log_dir: Path,
    full_log: bool = True,
) -> EpisodeResult:
    """The one function other lanes call. AIRTIGHT_ENGINE picks "stub" (the default) or "v0".

    full_log = False writes the smallest log the contract accepts, the header and the outcome,
    for searches that run thousands of episodes. The stub's log is already that small.
    """
    engine = os.environ.get(ENGINE_ENV, "stub")
    if engine == "stub":
        return _run_stub(site, fleet, tactic, sensor_curves, seed, log_dir)
    if engine == "v0":
        return _run_v0(site, fleet, tactic, sensor_curves, seed, log_dir, full_log)
    raise ValueError(f"{ENGINE_ENV}={engine!r} is not a valid engine; choose one of {ENGINES}")


def _run_v0(
    site: Site,
    fleet: FleetConfig,
    tactic: Tactic,
    sensor_curves: SensorCurves,
    seed: int,
    log_dir: Path,
    full_log: bool,
) -> EpisodeResult:
    """The real engine, battery and phase on. The verdict is the scores read at TAU_REF."""
    header = EpisodeHeader(
        site_hash=site.content_hash(),
        fleet_hash=fleet.content_hash(),
        sensor_curve_hash=sensor_curves.content_hash(),
        tactic=tactic,
        seed=seed,
        sim_version=SIM_VERSION_V0,
    )
    events: list[BaseModel]
    params = EpisodeParams(battery=True)
    if full_log:
        recorder = LogRecorder(dt=params.dt)
        scores = simulate(site, fleet, tactic, sensor_curves, seed, params, recorder)
        events = recorder.events
    else:
        scores = simulate(site, fleet, tactic, sensor_curves, seed, params)
        events = [outcome_event(scores)]
    outcome = events[-1]
    assert isinstance(outcome, OutcomeEvent)
    log_path = log_dir / f"{fleet.name}__{tactic.id}__{seed}.jsonl"
    write_episode_log(log_path, header, events)
    return EpisodeResult(
        timely_detected=outcome.timely_detected,
        t_alarm=outcome.t_alarm,
        t_cdp=outcome.t_cdp,
        log_path=log_path,
    )


def _run_stub(
    site: Site,
    fleet: FleetConfig,
    tactic: Tactic,
    sensor_curves: SensorCurves,
    seed: int,
    log_dir: Path,
) -> EpisodeResult:
    """Hour-1 stub: a random outcome with the real shape and a valid log. Lane B replaces this at H8."""
    rng = np.random.default_rng([seed, int(tactic.content_hash(), 16)])
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
