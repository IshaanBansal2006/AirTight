from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

from airtight.contracts import (
    XY,
    BenignRoute,
    FixedSensor,
    FleetConfig,
    SensorCurves,
    Site,
    Tactic,
    read_episode_log,
)
from airtight.sim import episode as episode_module
from airtight.sim.constants import NEVER_SEEN, TAU_REF
from airtight.sim.episode import EpisodeParams, EpisodeScores, simulate
from airtight.sim.recorder import timely_at_ref
from airtight.sim.runner import ENGINE_ENV, run_episode

if TYPE_CHECKING:
    from pathlib import Path

FleetOf = Callable[[int], FleetConfig]
WEST_CAMERA = FixedSensor(
    id="cam_west", position=XY(x=60, y=100), sensor_type="fixed_cam", heading_deg=180
)


class Tape:
    """A recorder that just remembers what it was told."""

    def __init__(self) -> None:
        self.pose_times: list[float] = []
        self.look_times: list[float] = []
        self.finished: EpisodeScores | None = None

    def on_poses(self, t: float, poses: Any) -> None:
        self.pose_times.append(t)

    def on_looks(self, t: float, looks: Any) -> None:
        self.look_times.extend(t for _ in looks)

    def on_finish(self, scores: EpisodeScores) -> None:
        self.finished = scores


def test_params_are_frozen_and_hashable() -> None:
    assert hash(EpisodeParams()) == hash(EpisodeParams())
    assert EpisodeParams(dt=0.5) != EpisodeParams()


def test_same_seed_equal_scores_other_seed_differs(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    run = partial(simulate, yard_site, fleet_of(2), yard_tactic, yard_curve)
    assert run(1000) == run(1000)
    assert any(run(1000).n_looks != run(seed).n_looks for seed in (1001, 1002, 1003))


def test_scores_shape_and_timeline(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    scores = simulate(yard_site, fleet_of(4), yard_tactic, yard_curve, 1000)
    assert scores.seed == 1000
    assert scores.t_reach == pytest.approx(130.0 / 2.5)
    assert scores.t_cdp == pytest.approx(scores.t_reach - 25.0)
    assert scores.t_end == scores.t_reach + 10.0
    assert scores.sim_hours == scores.t_end / 3600.0
    assert scores.decoy_peak is None
    assert "intruder" not in scores.benign_peaks and "decoy" not in scores.benign_peaks


def test_t_zero_is_hit_exactly_and_nothing_happens_before_it(
    yard_site: Site,
    yard_curve: SensorCurves,
    yard_tactic: Tactic,
    fleet_of: FleetOf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    look_pass_times: list[float] = []
    real = episode_module.do_looks

    def spy(observers: Any, objects: Any, t: float, *rest: Any) -> Any:
        look_pass_times.append(t)
        return real(observers, objects, t, *rest)

    monkeypatch.setattr(episode_module, "do_looks", spy)
    tape = Tape()
    # an awkward warm-up that is not a whole number of steps still lands on t = 0 exactly
    params = EpisodeParams(warmup_s=37.1)
    scores = simulate(yard_site, fleet_of(4), yard_tactic, yard_curve, 1000, params, tape)
    assert look_pass_times[0] == 0.0 and min(look_pass_times) == 0.0
    assert tape.pose_times[0] == 0.0 and min(tape.pose_times) == 0.0
    assert all(t >= 0.0 for t in tape.look_times)
    assert look_pass_times[-1] <= scores.t_end
    assert tape.finished == scores and len(tape.look_times) == scores.n_looks


def test_recorder_does_not_change_the_result(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    run = partial(simulate, yard_site, fleet_of(2), yard_tactic, yard_curve, 1000)
    assert run() == run(recorder=Tape())


def test_process_pool_matches_serial(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    seeds = [1000, 1001, 1002, 1003]
    run = partial(simulate, yard_site, fleet_of(2), yard_tactic, yard_curve)
    serial = [run(seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = list(pool.map(run, seeds))
    assert pooled == serial


def test_parked_agent_far_from_the_path_never_sees_the_intruder(
    yard_site: Site,
    yard_curve: SensorCurves,
    yard_tactic: Tactic,
    fleet_of: Callable[..., FleetConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parked = fleet_of(1, speed_mps=0.001)
    scores = simulate(yard_site, parked, yard_tactic, yard_curve, 1000)
    assert scores.intruder_peak == NEVER_SEEN and scores.intruder_t_alarm_ref is None
    monkeypatch.setenv(ENGINE_ENV, "v0")
    result = run_episode(yard_site, parked, yard_tactic, yard_curve, 1000, tmp_path)
    assert result.timely_detected is False and result.t_alarm is None


def test_fixed_camera_on_the_entry_makes_detection_more_frequent(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    watched = yard_site.model_copy(update={"fixed_sensors": [WEST_CAMERA]})
    seeds = range(1000, 1020)
    without = sum(
        timely_at_ref(simulate(yard_site, fleet_of(1), yard_tactic, yard_curve, s)) for s in seeds
    )
    with_camera = sum(
        timely_at_ref(simulate(watched, fleet_of(1), yard_tactic, yard_curve, s)) for s in seeds
    )
    assert with_camera > without


def test_setup_problems_are_all_reported_at_once(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    llama = BenignRoute(
        id="llama", cls="llama", waypoints=[XY(x=30, y=30), XY(x=60, y=30)], arrival_rate_per_hour=1
    )
    radar = FixedSensor(id="d0", position=XY(x=60, y=100), sensor_type="radar")
    broken = yard_site.model_copy(
        update={"benign_routes": [*yard_site.benign_routes, llama], "fixed_sensors": [radar]}
    )
    with pytest.raises(ValueError) as err:
        simulate(broken, fleet_of(1), yard_tactic, yard_curve, 1, EpisodeParams(dt=1.0))
    message = str(err.value)
    assert "'llama'" in message  # a benign class with no false-positive rate
    assert "'radar'" in message  # a sensor type with no curve
    assert "'d0'" in message  # an id shared by an agent and a fixed sensor
    assert "shorter than dt" in message  # a 2 Hz sensor cannot look every 1 s step


def test_timely_verdict_equals_peak_at_cdp_over_threshold(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    verdicts = []
    for n in (1, 4):
        for seed in range(1000, 1012):
            scores = simulate(yard_site, fleet_of(n), yard_tactic, yard_curve, seed)
            assert timely_at_ref(scores) == (scores.intruder_peak >= TAU_REF)
            verdicts.append(timely_at_ref(scores))
    assert True in verdicts and False in verdicts  # the check saw both outcomes


def test_v0_log_reads_back_and_small_log_is_smaller(
    yard_site: Site,
    yard_curve: SensorCurves,
    yard_tactic: Tactic,
    fleet_of: FleetOf,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENGINE_ENV, "v0")
    fleet = fleet_of(4)
    watched = yard_site.model_copy(update={"fixed_sensors": [WEST_CAMERA]})
    full = run_episode(watched, fleet, yard_tactic, yard_curve, 1000, tmp_path / "full")
    small = run_episode(
        watched, fleet, yard_tactic, yard_curve, 1000, tmp_path / "small", full_log=False
    )
    assert (full.timely_detected, full.t_alarm, full.t_cdp) == (
        small.timely_detected,
        small.t_alarm,
        small.t_cdp,
    )

    header, event_iter = read_episode_log(full.log_path)
    events = list(event_iter)
    assert header.sim_version == "v0" and header.seed == 1000
    assert header.site_hash == watched.content_hash()
    assert all(e.t >= 0.0 for e in events)
    assert [e.t for e in events] == sorted(e.t for e in events)
    kinds = {e.kind for e in events}
    assert {"position", "score", "outcome"} <= kinds
    assert kinds <= {"position", "detection", "score", "alarm_delivered", "outcome"}
    assert events[-1].kind == "outcome"
    assert events[-1].timely_detected == full.timely_detected
    intruder_poses = [e for e in events if e.kind == "position" and e.object_id == "intruder"]
    assert (intruder_poses[0].position.x, intruder_poses[0].position.y) == (20.0, 100.0)
    assert [e.t for e in intruder_poses[:3]] == [0.0, 0.5, 1.0]  # 2 Hz
    alarms = [e for e in events if e.kind == "alarm_delivered"]
    assert [a.t for a in alarms] == ([full.t_alarm] if full.t_alarm is not None else [])

    _, small_events = read_episode_log(small.log_path)
    assert [e.kind for e in small_events] == ["outcome"]
    assert small.log_path.stat().st_size < full.log_path.stat().st_size / 10


def test_engine_selection(
    yard_site: Site,
    yard_curve: SensorCurves,
    yard_tactic: Tactic,
    fleet_of: FleetOf,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENGINE_ENV, raising=False)
    default = run_episode(yard_site, fleet_of(2), yard_tactic, yard_curve, 7, tmp_path / "a")
    header, event_iter = read_episode_log(default.log_path)
    assert [e.kind for e in event_iter] == ["outcome"]  # consuming it closes the file
    assert header.sim_version == "stub-0"  # the stub stays the default
    monkeypatch.setenv(ENGINE_ENV, "warp")
    with pytest.raises(ValueError, match="stub.*v0"):
        run_episode(yard_site, fleet_of(2), yard_tactic, yard_curve, 7, tmp_path / "b")


def test_two_drone_episode_takes_under_a_second(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    simulate(yard_site, fleet_of(2), yard_tactic, yard_curve, 999)  # warm imports and caches
    start = time.perf_counter()
    simulate(yard_site, fleet_of(2), yard_tactic, yard_curve, 1000)
    assert time.perf_counter() - start < 1.0


def test_task_time_default_reproduces_todays_numbers_exactly(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    from airtight.sim import adapt

    assert EpisodeParams().task_time_s == 0.0
    for seed in (1000, 1001, 1002):
        default = simulate(yard_site, fleet_of(2), yard_tactic, yard_curve, seed)
        explicit = simulate(
            yard_site, fleet_of(2), yard_tactic, yard_curve, seed, EpisodeParams(task_time_s=0.0)
        )
        assert default == explicit
        assert default.t_cdp == adapt.t_cdp(yard_site, yard_tactic)  # the contract's definition
        assert default.t_end == default.t_reach + EpisodeParams().tail_s


def test_task_time_moves_the_cdp_and_never_lowers_the_timely_fraction(
    yard_site: Site, yard_curve: SensorCurves, fleet_of: FleetOf
) -> None:
    from airtight.sim import scenarios

    seeds = range(1000, 1016)
    what_if = EpisodeParams(task_time_s=60.0)
    for tactic_name in ("jog", "sprint"):
        tactic = scenarios.load_tactic(tactic_name)
        base = [simulate(yard_site, fleet_of(2), tactic, yard_curve, s) for s in seeds]
        late = [simulate(yard_site, fleet_of(2), tactic, yard_curve, s, what_if) for s in seeds]
        for a, b in zip(base, late, strict=True):
            assert b.t_reach == a.t_reach
            assert b.t_cdp == max(a.t_reach + 60.0 - 25.0, 0.0) and b.t_end == a.t_end + 60.0
        assert sum(map(timely_at_ref, late)) >= sum(map(timely_at_ref, base))
    # the sprint cannot be caught in time without a task time, and can be with one
    assert sum(map(timely_at_ref, base)) == 0 and sum(map(timely_at_ref, late)) > 0


def test_run_episode_never_uses_the_task_time_what_if(
    yard_site: Site,
    yard_curve: SensorCurves,
    yard_tactic: Tactic,
    fleet_of: FleetOf,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from airtight.sim import adapt

    monkeypatch.setenv(ENGINE_ENV, "v0")
    result = run_episode(yard_site, fleet_of(2), yard_tactic, yard_curve, 1000, tmp_path)
    assert result.t_cdp == adapt.t_cdp(yard_site, yard_tactic)


def test_weight_mode_changes_the_patrol_and_bad_mode_is_rejected(
    yard_site: Site, yard_curve: SensorCurves, yard_tactic: Tactic, fleet_of: FleetOf
) -> None:
    run = partial(simulate, yard_site, fleet_of(2), yard_tactic, yard_curve)
    by_mode = {
        m: [run(s, EpisodeParams(weight_mode=m)).n_looks for s in range(1000, 1006)]
        for m in ("asset", "uniform", "band")
    }
    assert by_mode["asset"] != by_mode["uniform"] and by_mode["asset"] != by_mode["band"]
    with pytest.raises(ValueError, match="spiral"):
        run(1000, EpisodeParams(weight_mode="spiral"))
