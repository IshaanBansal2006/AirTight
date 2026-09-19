"""Approximate policy iteration for the learned CBBA score.

One ROLLOUT is a simulator-free mission: the fleet takes off at the scene's
start poses, CBBA allocates a fixed set of goto tasks once, the executor flies
them on the SmoothBackend, and each drone runs its ego filter on a synthesised
IMU plus sign fixes from its mounted camera (signs only — vehicle landmarks are
skipped for speed; the return does not depend on them). The RETURN is

    sum over completed tasks of  reward * quality(pose variance at completion)

with `quality` from learned_score.py: a task done while badly localised counts
for less. No time discount — the horizon is the pressure.

CREDIT is counterfactual against the previous iterate, under common random
numbers: for drone d, credit_d = R(all drones on the new score) - R(d on the
previous score, others on the new score), both rollouts with the same seed so
the only difference is d's bids. That is d's marginal contribution, coupling
included (its bids change everyone's bundles).

ITERATION: fit a ridge regression of credit on the score's features; add the
fitted weights to the current ones (the credit is an improvement signal
relative to the previous iterate, so at a fixed point it vanishes and the
weights stop moving). Evaluate on held-out seeds against the previous
iterate and the frozen classical score; record the fraction of allocations
that converged before CBBA's round cap; stop on no improvement, a convergence
collapse, or the iteration budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from airtight.swarm.autonomy.allocator import CBBAAllocator, PathScore
from airtight.swarm.autonomy.decomposer import Task
from airtight.swarm.autonomy.executor import Executor, SmoothBackend
from airtight.swarm.autonomy.learned_score import FEATURE_NAMES, LearnedScore, quality
from airtight.swarm.autonomy.world_state import DroneState, WorldState
from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import EgoConfig, ImuConfig
from airtight.swarm.edge.ego import EgoFilter
from airtight.swarm.edge.imu import ImuSynthesizer
from airtight.swarm.edge.pipeline import road_config
from airtight.swarm.edge.sensing import MeasurementSynthesizer, PlatformTruth
from airtight.swarm.scene import Scene, demo_scene

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

log = logging.getLogger(__name__)

CONTROL_DT = 0.1


@dataclass
class MissionSpec:
    seed: int
    tasks: list[Task]


@dataclass
class Completion:
    task_id: str
    drone_id: str
    t: float
    pose_var: float
    reward: float


@dataclass
class MissionResult:
    total: float
    per_drone: dict[str, float]
    completions: list[Completion]
    converged: bool
    paths: dict[str, list[Task]]


@dataclass
class TrainConfig:
    n_tasks: int = 6
    horizon_s: float = 60.0
    speed: float = 4.0
    imu_rate_hz: float = 50.0
    train_missions: int = 8
    holdout_missions: int = 4
    max_iters: int = 4
    ridge: float = 1.0
    min_improvement: float = 0.0  # held-out mean return over the previous iterate
    min_convergence_rate: float = 0.8
    explore_std: float = 0.1  # bootstrap: iteration 1 starts from a perturbed score (see train)
    s0: float = 1.0
    task_altitude: float = 4.0
    seed: int = 0


def random_tasks(rng: np.random.Generator, scene: Scene, n: int, altitude: float) -> list[Task]:
    """Goto tasks scattered over the patrol area, rewards in [0.5, 2]."""
    xs, ys = zip(*scene.patrol_area, strict=False)
    tasks = []
    for i in range(n):
        wp = [float(rng.uniform(min(xs), max(xs))), float(rng.uniform(min(ys), max(ys))), altitude]
        tasks.append(
            Task(
                task_id=f"g{i}",
                intent_id="train",
                task_type="goto_waypoint",
                waypoints=[wp],
                reward=float(rng.uniform(0.5, 2.0)),
            )
        )
    return tasks


class MissionSim:
    def __init__(self, scene: Scene, cfg: TrainConfig) -> None:
        self.scene = scene
        self.cfg = cfg
        self.drone_ids = sorted(scene.fleet)
        self.sensor_cfg = road_config(scene, self.drone_ids)
        self.imu_cfg = ImuConfig(rate_hz=cfg.imu_rate_hz)
        self.ego_cfg = EgoConfig(imu=self.imu_cfg)

    def run(self, spec: MissionSpec, scores: dict[str, PathScore]) -> MissionResult:
        """One rollout; `scores` maps drone_id -> the score that drone bids with."""
        rng = np.random.default_rng(spec.seed)  # common random numbers: seed is the spec's
        world = WorldState()
        egos: dict[str, EgoFilter] = {}
        imus: dict[str, ImuSynthesizer] = {}
        for i, did in enumerate(self.drone_ids):
            start = self.scene.fleet[did]
            d = DroneState(
                drone_id=did,
                position=list(start.position),
                orientation=rotation.from_yaw(start.yaw).tolist(),
                speed=self.cfg.speed,
            )
            world.update_drone(d)
            egos[did] = EgoFilter(
                self.ego_cfg,
                self.scene.signs,
                np.asarray(d.position),
                np.zeros(3),
                np.asarray(d.orientation),
            )
            imus[did] = ImuSynthesizer(self.imu_cfg, np.random.default_rng(spec.seed * 101 + i))
            d.pose_estimate = egos[did].to_msg(did, 0.0)
        synth = MeasurementSynthesizer(self.scene, self.sensor_cfg, rng)

        any_score = next(iter(scores.values()))
        allocator = CBBAAllocator(score=any_score, score_overrides=scores)
        paths = allocator.allocate(spec.tasks, world.available_drones())
        backend = SmoothBackend(world)
        executor = Executor(backend, world)
        executor.assign(paths)

        dt_imu = self.imu_cfg.dt
        steps = int(round(CONTROL_DT / dt_imu))
        for did in self.drone_ids:
            d = world.drones[did]
            imus[did].sample(0.0, np.asarray(d.velocity), np.asarray(d.orientation))
        R = {
            sid: np.diag(np.square(np.asarray(s.measurement_noise)))
            for sid, s in self.sensor_cfg.sensors.items()
        }
        completions: list[Completion] = []
        t = 0.0
        n_ticks = int(round(self.cfg.horizon_s / CONTROL_DT))
        for _ in range(n_ticks):
            before = len(executor.completed)
            current = {did: q.current for did, q in executor._queues.items()}
            # sub-step the dynamics at the IMU rate, then let the BTs tick once
            for _ in range(steps - 1):
                backend.step(dt_imu)
                t += dt_imu
                self._imu_step(t, world, imus, egos)
            executor.tick(dt_imu)
            t += dt_imu
            self._imu_step(t, world, imus, egos)
            platforms = {
                did: PlatformTruth(
                    np.asarray(world.drones[did].position),
                    np.asarray(world.drones[did].orientation),
                )
                for did in self.drone_ids
            }
            dets, _ = synth.observe(t, platforms, [])
            for did in self.drone_ids:
                sid = f"cam_{did}"
                mine = [x for x in dets if x.sensor_id == sid and x.class_label == "sign"]
                cam_cfg = self.sensor_cfg.sensors[sid]
                egos[did].update(mine, cam_cfg, t, R[sid])  # type: ignore[arg-type]
            for task_id in executor.completed[before:]:
                did = next(
                    d for d, task in current.items() if task is not None and task.task_id == task_id
                )
                var = float(np.trace(egos[did].P[0:3, 0:3]))
                reward = next(task.reward for task in spec.tasks if task.task_id == task_id)
                completions.append(Completion(task_id, did, t, var, reward))
            if executor.idle():
                break
        per_drone = dict.fromkeys(self.drone_ids, 0.0)
        for c in completions:
            per_drone[c.drone_id] += c.reward * float(quality(c.pose_var, self.cfg.s0))
        return MissionResult(
            total=float(sum(per_drone.values())),
            per_drone=per_drone,
            completions=completions,
            converged=allocator.last_converged,
            paths=paths,
        )

    @staticmethod
    def _imu_step(
        t: float, world: WorldState, imus: dict[str, ImuSynthesizer], egos: dict[str, EgoFilter]
    ) -> None:
        for did, imu in imus.items():
            d = world.drones[did]
            s = imu.sample(t, np.asarray(d.velocity), np.asarray(d.orientation))
            if s is not None:
                egos[did].predict(s)


# ------------------------------------------------------------------ training
@dataclass
class IterationStats:
    iteration: int
    weights: list[float]
    train_return: float
    holdout_return: float
    holdout_return_previous: float
    holdout_return_classical: float
    convergence_rate: float
    n_samples: int


@dataclass
class TrainingReport:
    iterations: list[IterationStats] = field(default_factory=list)
    stopped_because: str = ""
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))


def counterfactual_credit(
    sim: MissionSim, spec: MissionSpec, new: LearnedScore, prev: LearnedScore
) -> tuple[MissionResult, dict[str, float]]:
    """R(all new) and, per drone, R(all new) - R(that drone on prev), same seed."""
    all_new = sim.run(spec, dict.fromkeys(sim.drone_ids, new))
    credit: dict[str, float] = {}
    for did in sim.drone_ids:
        scores: dict[str, PathScore] = dict.fromkeys(sim.drone_ids, new)
        scores[did] = prev
        credit[did] = all_new.total - sim.run(spec, scores).total
    return all_new, credit


def _fit_ridge(phi: NDArray[np.float64], y: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    A = phi.T @ phi + lam * np.eye(phi.shape[1])
    return np.asarray(np.linalg.solve(A, phi.T @ y), dtype=float)


def train(
    scene: Scene | None = None,
    cfg: TrainConfig | None = None,
    log_fn: Callable[[str], object] = print,
) -> tuple[LearnedScore, TrainingReport]:
    scene = scene or demo_scene()
    cfg = cfg or TrainConfig()
    sim = MissionSim(scene, cfg)
    rng = np.random.default_rng(cfg.seed)
    train_specs = [
        MissionSpec(
            seed=int(rng.integers(1 << 30)),
            tasks=random_tasks(rng, scene, cfg.n_tasks, cfg.task_altitude),
        )
        for _ in range(cfg.train_missions)
    ]
    holdout = [
        MissionSpec(
            seed=int(rng.integers(1 << 30)),
            tasks=random_tasks(rng, scene, cfg.n_tasks, cfg.task_altitude),
        )
        for _ in range(cfg.holdout_missions)
    ]
    classical = LearnedScore.zero(scene.signs, s0=cfg.s0)
    prev = LearnedScore.zero(scene.signs, s0=cfg.s0)
    # Bootstrap. Credit is measured against the PREVIOUS iterate under common
    # random numbers, so two identical policies give identical rollouts and a
    # credit of exactly zero (the CRN test pins this) — starting current == prev
    # would fit zero forever (the first training run did exactly that: four
    # iterations, weights all zero). Iteration 1 therefore starts from a small
    # random perturbation of the classical score; from iteration 2 the loop is
    # .
    current = LearnedScore(
        classical.predictor, rng.normal(0.0, cfg.explore_std, len(FEATURE_NAMES)), s0=cfg.s0
    )
    report = TrainingReport()

    def evaluate(score: LearnedScore) -> tuple[float, float]:
        results = [sim.run(spec, dict.fromkeys(sim.drone_ids, score)) for spec in holdout]
        return (
            float(np.mean([r.total for r in results])),
            float(np.mean([r.converged for r in results])),
        )

    classical_holdout, _ = evaluate(classical)
    prev_holdout = classical_holdout
    for it in range(1, cfg.max_iters + 1):
        rows, targets, totals, conv = [], [], [], []
        for spec in train_specs:
            result, credit = counterfactual_credit(sim, spec, current, prev)
            totals.append(result.total)
            conv.append(result.converged)
            for did, path in result.paths.items():
                drone = DroneState(drone_id=did, position=list(scene.fleet[did].position))
                rows.append(current.features(drone, path))
                targets.append(credit[did])
        phi, y = np.asarray(rows), np.asarray(targets)
        delta = _fit_ridge(phi, y, cfg.ridge)
        candidate = LearnedScore(current.predictor, current.weights + delta, s0=cfg.s0)
        holdout_return, conv_rate = evaluate(candidate)
        stats = IterationStats(
            it,
            candidate.weights.tolist(),
            float(np.mean(totals)),
            holdout_return,
            prev_holdout,
            classical_holdout,
            conv_rate,
            len(rows),
        )
        report.iterations.append(stats)
        log_fn(
            f"iter {it}: holdout {holdout_return:.3f} (prev {prev_holdout:.3f}, classical "
            f"{classical_holdout:.3f}) convergence {conv_rate:.2f} train {np.mean(totals):.3f}"
        )
        if conv_rate < cfg.min_convergence_rate:
            report.stopped_because = "convergence rate collapsed"
            break
        if holdout_return - prev_holdout < cfg.min_improvement:
            report.stopped_because = "no held-out improvement over the previous iterate"
            break
        prev, current, prev_holdout = current, candidate, holdout_return
    else:
        report.stopped_because = "iteration budget"
    return current, report
