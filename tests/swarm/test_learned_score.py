"""Learned CBBA path score (decisions 023/024): the pluggable seam, the
localization predictor, the DMG property of the baseline, common random
numbers in the counterfactual credit, and one tiny policy-iteration step."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.autonomy.allocator import CBBAAllocator, TimeDiscountedScore
from airtight.swarm.autonomy.decomposer import Task
from airtight.swarm.autonomy.learned_score import (
    FEATURE_NAMES,
    LearnedScore,
    LocalizationPredictor,
    quality,
)
from airtight.swarm.autonomy.training import (
    MissionSim,
    MissionSpec,
    TrainConfig,
    counterfactual_credit,
    random_tasks,
    train,
)
from airtight.swarm.autonomy.world_state import DroneState
from airtight.swarm.scene import demo_scene

if TYPE_CHECKING:
    from pathlib import Path


def goto(tid: str, x: float, y: float, reward: float = 1.0) -> Task:
    return Task(
        task_id=tid,
        intent_id="i",
        task_type="goto_waypoint",
        waypoints=[[x, y, 4.0]],
        reward=reward,
    )


def _best_insert_gain(
    score: TimeDiscountedScore, drone: DroneState, path: list[Task], task: Task
) -> float:
    base = score.score(drone, path)
    return (
        max(score.score(drone, path[:k] + [task] + path[k:]) for k in range(len(path) + 1)) - base
    )


def test_baseline_score_dmg_holds_for_append_but_not_for_best_insertion() -> None:
    """CBBA's convergence proof needs diminishing marginal gain (DMG): adding a
    task elsewhere never makes another task's gain LARGER. The time-discounted
    score has it when tasks are APPENDED (the proof's setting) — but the
    allocator inserts at the best position, and there a task already in the
    path can act as a paid-for detour into a neighbour's area, RAISING that
    neighbour's gain. So the shipped CBBA never had the guarantee strictly;
    the learned score loses less than it says. Both facts are pinned here."""
    rng = np.random.default_rng(0)
    score = TimeDiscountedScore()
    drone = DroneState(drone_id="d", position=[0.0, 0.0, 4.0])
    violations = 0
    for _ in range(300):
        tasks = [
            goto(f"t{i}", *rng.uniform(-30, 30, 2), reward=rng.uniform(0.5, 2)) for i in range(4)
        ]
        a, b, rest = tasks[0], tasks[1], tasks[2:]
        # append-only: DMG must hold exactly
        gain_b = score.score(drone, rest + [b]) - score.score(drone, rest)
        gain_b_after_a = score.score(drone, rest + [a, b]) - score.score(drone, rest + [a])
        assert gain_b_after_a <= gain_b + 1e-9
        # best-position insertion (what CBBAAllocator does): can violate
        with_a = max(
            (rest[:k] + [a] + rest[k:] for k in range(3)), key=lambda p: score.score(drone, p)
        )
        if (
            _best_insert_gain(score, drone, with_a, b)
            > _best_insert_gain(score, drone, rest, b) + 1e-9
        ):
            violations += 1
    assert violations > 0, "expected best-insertion DMG violations; the docs claim there are some"
    assert violations < 150  # ...but it is the minority case


def test_allocator_default_score_is_unchanged_and_reports_convergence() -> None:
    drones = [
        DroneState(drone_id="d0", position=[0, 0, 4]),
        DroneState(drone_id="d1", position=[40, 0, 4]),
    ]
    tasks = [goto("a", 5, 0), goto("b", 35, 0), goto("c", 20, 10)]
    alloc = CBBAAllocator().allocate(tasks, drones)
    assert "a" in [t.task_id for t in alloc["d0"]] and "b" in [t.task_id for t in alloc["d1"]]
    a = CBBAAllocator()
    a.allocate(tasks, drones)
    assert a.last_converged and a.last_rounds >= 1


def test_predictor_drifts_with_distance_and_resets_at_signs() -> None:
    scene = demo_scene()
    pred = LocalizationPredictor(scene.signs, drift_rate=0.02, sign_range=12.0, fix_floor=0.01)
    drone = DroneState(drone_id="d", position=[-30.0, 30.0, 4.0])
    far = goto("far", -30.0, 80.0)  # 30+ m from every sign: pure dead reckoning
    var, passes = pred.along_path(drone, [far])
    assert passes == 0 and abs(var[0] - (0.01 + 0.02 * 50.0)) < 1e-9
    drone = DroneState(drone_id="d", position=[-8.0, 3.0, 4.0])
    along = goto("road", 30.0, 3.0)  # down the road past s0 and s2
    var2, passes2 = pred.along_path(drone, [along])
    assert passes2 >= 1 and var2[0] == 0.01
    assert quality(0.0, 1.0) == 1.0 and abs(quality(1.0, 1.0) - 0.5) < 1e-12


def test_zero_weights_equal_baseline_and_json_roundtrip(tmp_path: Path) -> None:
    scene = demo_scene()
    zero = LearnedScore.zero(scene.signs)
    drone = DroneState(drone_id="d", position=[0.0, 0.0, 4.0])
    path = [goto("a", 10, 0), goto("b", 20, 5)]
    assert abs(zero.score(drone, path) - TimeDiscountedScore().score(drone, path)) < 1e-12
    learned = LearnedScore(zero.predictor, np.arange(len(FEATURE_NAMES), dtype=float) * 0.1, s0=1.5)
    learned.to_json(tmp_path / "w.json")
    back = LearnedScore.from_json(tmp_path / "w.json", scene.signs)
    assert abs(back.score(drone, path) - learned.score(drone, path)) < 1e-12
    bad = json.loads((tmp_path / "w.json").read_text())
    bad["features"] = ["x"]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    try:
        LearnedScore.from_json(tmp_path / "bad.json", scene.signs)
    except ValueError as e:
        assert "retrain" in str(e)
    else:
        raise AssertionError("mismatched feature layout must be refused")


def small_cfg() -> TrainConfig:
    return TrainConfig(
        n_tasks=2,
        horizon_s=12.0,
        imu_rate_hz=20.0,
        train_missions=2,
        holdout_missions=1,
        max_iters=1,
    )


def test_counterfactual_credit_is_exactly_zero_under_common_random_numbers() -> None:
    """If the new score IS the previous one, the counterfactual rollouts are
    identical and the credit is 0 to the bit — which is only true if the seed
    controls every random draw in the rollout."""
    scene = demo_scene()
    sim = MissionSim(scene, small_cfg())
    spec = MissionSpec(seed=3, tasks=random_tasks(np.random.default_rng(3), scene, 3, 4.0))
    zero = LearnedScore.zero(scene.signs)
    result, credit = counterfactual_credit(sim, spec, zero, zero)
    assert result.completions, "the tiny mission must complete something"
    assert all(c == 0.0 for c in credit.values()), credit


def test_return_weights_rewards_by_localization_quality() -> None:
    scene = demo_scene()
    sim = MissionSim(scene, small_cfg())
    spec = MissionSpec(seed=5, tasks=random_tasks(np.random.default_rng(5), scene, 3, 4.0))
    r = sim.run(spec, dict.fromkeys(sim.drone_ids, LearnedScore.zero(scene.signs)))
    raw = sum(c.reward for c in r.completions)
    assert 0.0 < r.total <= raw + 1e-12
    assert all(c.pose_var >= 0.0 for c in r.completions)


def test_one_policy_iteration_runs_and_reports() -> None:
    score, report = train(demo_scene(), small_cfg(), log_fn=lambda *_: None)
    assert len(report.iterations) == 1
    it = report.iterations[0]
    assert it.n_samples == 2 * 2  # missions x drones
    assert np.all(np.isfinite(it.weights)) and np.isfinite(it.holdout_return)
    # the first full run fitted exactly zero for four iterations (identical start
    # and reference policies -> zero credit under common random numbers); a
    # bootstrapped iteration must leave the weights nonzero
    assert np.abs(it.weights).max() > 0.0
    assert np.abs(np.asarray(it.weights)).sum() != 0.0
    assert 0.0 <= it.convergence_rate <= 1.0
    assert report.stopped_because
