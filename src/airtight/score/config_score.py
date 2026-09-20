"""One fleet configuration's score: operating point, detection, intervals, alert rates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from airtight.score import roc
from airtight.sim.constants import TAU_INVESTIGATE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from airtight.sim.episode import EpisodeScores, QuietScores

    Interval = tuple[float, float]

DECOY_ID = "decoy"


@dataclass(frozen=True)
class RocRow:
    tau: float
    pd: float
    pd_ci: Interval  # seeds resampled, tau held fixed
    far: float


@dataclass(frozen=True)
class ConfigScore:
    tau: float  # the operating threshold
    flag: str  # roc.FLAG_OK, FLAG_FAR_LIMITED or FLAG_FAR_FLOOR
    far: float  # false alarms per hour at tau
    pd: float  # mean over tactics, every tactic weighing the same
    pd_ci: Interval
    pd_by_tactic: dict[str, float]
    worst_tactic_id: str
    worst_tactic_pd: float
    worst_tactic_pd_ci: Interval
    roc: tuple[RocRow, ...]
    human_decisions_per_hour: float  # the false alarm rate at tau
    raw_alerts_per_hour: float  # the rate at tau_min: what an operator sees with no filtering
    n_seeds: int
    quiet_hours: float


def peaks_and_quiet(
    episodes_by_tactic: Mapping[str, Sequence[EpisodeScores]], quiet_runs: Sequence[QuietScores]
) -> tuple[list[str], npt.NDArray[np.float64], list[tuple[npt.NDArray[np.float64], float]]]:
    """Check the inputs and lay them out for roc: tactic ids, peaks (n_seeds, n_tactics), quiet.

    Every tactic must have exactly the same seeds in the same order: the rows of the peaks
    matrix are seeds, and intervals come from resampling whole rows.
    """
    if not episodes_by_tactic:
        raise ValueError("score_config needs at least one tactic")
    if not quiet_runs:
        raise ValueError("score_config needs at least one quiet run for the false alarm rate")
    tactic_ids = list(episodes_by_tactic)
    seeds = [e.seed for e in episodes_by_tactic[tactic_ids[0]]]
    if not seeds:
        raise ValueError(f"tactic {tactic_ids[0]!r} has no episodes")
    for tactic_id in tactic_ids:
        mine = [e.seed for e in episodes_by_tactic[tactic_id]]
        if mine != seeds:
            raise ValueError(
                f"tactic {tactic_id!r} does not have the same seeds in the same order as "
                f"{tactic_ids[0]!r}: {len(mine)} episodes against {len(seeds)}"
                + ("" if len(mine) != len(seeds) else ", in a different order or with other seeds")
            )
        for episode in episodes_by_tactic[tactic_id]:
            assert DECOY_ID not in episode.benign_peaks, "the decoy must never count as benign"
    for run in quiet_runs:
        assert DECOY_ID not in run.benign_peaks, "the decoy must never count as benign"
    peaks = np.array(
        [[e.intruder_peak for e in episodes_by_tactic[t]] for t in tactic_ids], dtype=np.float64
    ).T
    quiet = [
        (np.array(list(run.benign_peaks.values()), dtype=np.float64), run.sim_hours)
        for run in quiet_runs
    ]
    return tactic_ids, peaks, quiet


def score_config(
    episodes_by_tactic: Mapping[str, Sequence[EpisodeScores]],
    quiet_runs: Sequence[QuietScores],
    far_target: float = 1.0,
    n_boot: int = 500,
    boot_seed: int = 0,
    tau_min: float = TAU_INVESTIGATE,
) -> ConfigScore:
    """Score one configuration from its intrusion episodes and its quiet nights.

    Every tactic must have exactly the same seeds in the same order: the rows of the peaks
    matrix are seeds, and intervals come from resampling whole rows.
    """
    tactic_ids, peaks, quiet = peaks_and_quiet(episodes_by_tactic, quiet_runs)

    op = roc.operating_point(peaks, quiet, far_target, tau_min)
    per_tactic, worst = roc.pd_by_tactic(peaks, op.tau)
    boot = roc.bootstrap(peaks, quiet, n_boot, boot_seed, far_target, tau_min)
    grid = roc.roc_grid(peaks, quiet, far_target=far_target, tau_min=tau_min)
    assert np.array_equal(grid.tau, boot.grid_tau) and boot.worst_index == worst
    rows = tuple(
        RocRow(float(t), float(p), (float(lo), float(hi)), float(f))
        for t, p, (lo, hi), f in zip(grid.tau, grid.pd, boot.grid_pd_ci, grid.far, strict=True)
    )
    return ConfigScore(
        tau=op.tau,
        flag=op.flag,
        far=op.far,
        pd=op.pd,
        pd_ci=boot.pd_ci,
        pd_by_tactic={t: float(v) for t, v in zip(tactic_ids, per_tactic, strict=True)},
        worst_tactic_id=tactic_ids[worst],
        worst_tactic_pd=float(per_tactic[worst]),
        worst_tactic_pd_ci=boot.worst_pd_ci,
        roc=rows,
        human_decisions_per_hour=op.far,
        raw_alerts_per_hour=float(roc.far_curve(quiet, np.array([tau_min]))[0]),
        n_seeds=peaks.shape[0],
        quiet_hours=float(sum(run.sim_hours for run in quiet_runs)),
    )
