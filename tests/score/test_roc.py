from __future__ import annotations

import numpy as np
import pytest

from airtight.score import roc
from airtight.score.roc import (
    FLAG_FAR_FLOOR,
    FLAG_FAR_LIMITED,
    FLAG_OK,
    bootstrap,
    candidates,
    far_curve,
    operating_point,
    pd_by_tactic,
    roc_grid,
    wilson_interval,
)
from airtight.sim.constants import NEVER_SEEN, TAU_INVESTIGATE


def _random_case(seed: int, n_seeds: int = 60, n_tactics: int = 4, n_quiet: int = 8):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    peaks = rng.normal(6.0, 5.0, size=(n_seeds, n_tactics))
    peaks[rng.random(peaks.shape) < 0.25] = NEVER_SEEN
    quiet = [(rng.normal(1.0, 2.5, size=int(rng.integers(2, 9))), 1.0) for _ in range(n_quiet)]
    return peaks, quiet


def test_pd_and_far_are_non_increasing_in_tau() -> None:
    peaks, quiet = _random_case(1)
    grid = roc_grid(peaks, quiet, n_points=1000)
    assert np.all(np.diff(grid.tau) > 0)
    assert np.all(np.diff(grid.pd) <= 0) and np.all(np.diff(grid.far) <= 0)
    assert grid.pd[0] <= 0.75 + 0.2  # a quarter of the cells were never seen at all


def test_hand_built_operating_point() -> None:
    # Ten benign peaks over 5 hours, target 1 false alarm per hour, so at most 5 benign objects
    # may sit at or above the threshold. Benign peaks: 10, 9, 8, 7, 6, | 5, 4, 3, 2.5, 2.
    #   tau = 5  -> six at or above -> 6 / 5 = 1.2 per hour: too many
    #   tau = 6  -> five at or above -> 5 / 5 = 1.0 per hour: allowed ("at most" includes equal)
    # The rule picks the LOWEST candidate that is allowed. With no intruder peak between 5 and 6
    # that is the fifth-highest benign peak itself, 6.0.
    benign = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0])
    quiet = [(benign[:4], 2.0), (benign[4:], 3.0)]  # hours are summed over the runs
    peaks = np.array([[9.5, 6.0], [7.5, 5.0], [4.5, NEVER_SEEN], [6.5, 3.0]])
    op = operating_point(peaks, quiet)
    assert (op.tau, op.far, op.flag) == (6.0, 1.0, FLAG_OK)
    # tactic 0: 9.5, 7.5, 6.5 of four -> 0.75. tactic 1: 6.0 only -> 0.25. Mean 0.5.
    assert op.pd == 0.5
    per_tactic, worst = pd_by_tactic(peaks, op.tau)
    assert per_tactic.tolist() == [0.75, 0.25] and worst == 1

    # An intruder peak between the sixth- and fifth-highest benign peaks is a lower candidate
    # with the same five benign objects above it, so the rule picks it instead.
    op2 = operating_point(np.array([[5.5, 6.0], [7.5, 5.0]]), quiet)
    assert (op2.tau, op2.far) == (5.5, 1.0)


def test_far_curve_counts_ties_and_sums_hours() -> None:
    quiet = [(np.array([4.0, 4.0, 2.0]), 1.5), (np.array([4.0]), 0.5)]
    assert far_curve(quiet, np.array([4.0, 4.0001, 2.0, 1.0])).tolist() == [1.5, 0.0, 2.0, 2.0]
    # two benign peaks equal to a candidate both count: 2 per hour at 4.0, so 4.0 is not allowed
    quiet2 = [(np.array([4.0, 4.0, 2.0]), 1.0)]
    op = operating_point(np.array([[4.0, 9.0]]), quiet2)
    assert op.tau == 9.0 and op.far == 0.0 and op.pd == 0.5


def test_edge_flags() -> None:
    peaks = np.array([[8.0, 3.0], [NEVER_SEEN, 5.0]])
    quiet_none = [(np.array([]), 2.0)]  # no benign peaks at all
    op = operating_point(peaks, quiet_none)
    assert (op.tau, op.flag, op.far) == (TAU_INVESTIGATE, FLAG_FAR_LIMITED, 0.0)
    assert op.pd == 0.75  # 8.0, 3.0 and 5.0 are at or above 1.5; NEVER_SEEN never is

    below = operating_point(peaks, [(np.array([0.5, -2.0, 1.4]), 2.0)])
    assert below.flag == FLAG_FAR_LIMITED and below.tau == TAU_INVESTIGATE

    # three benign objects share the highest peak of all, in half an hour: 6 per hour even there
    hot = [(np.array([9.0, 9.0, 9.0, 2.0]), 0.5)]
    floor = operating_point(peaks, hot)
    assert floor.flag == FLAG_FAR_FLOOR and floor.pd == 0.0 and floor.far == 0.0
    assert floor.tau > 9.0 and floor.tau == np.nextafter(9.0, np.inf)


def test_never_seen_is_never_a_candidate_and_never_detected() -> None:
    peaks = np.full((5, 2), NEVER_SEEN)
    quiet = [(np.array([NEVER_SEEN, 3.0]), 1.0)]
    assert candidates(peaks, quiet).tolist() == [TAU_INVESTIGATE, 3.0]
    # even with a floor below NEVER_SEEN, the sentinel itself is never a threshold
    assert candidates(peaks, quiet, tau_min=-2e9).tolist() == [-2e9, 3.0]
    per_tactic, _ = pd_by_tactic(peaks, -2e9)  # even a threshold below NEVER_SEEN detects nothing
    assert per_tactic.tolist() == [0.0, 0.0]
    assert operating_point(peaks, quiet).pd == 0.0


def test_roc_grid_always_holds_tau_min_and_the_operating_point() -> None:
    peaks, quiet = _random_case(2, n_seeds=200, n_tactics=6)
    op = operating_point(peaks, quiet)
    assert len(candidates(peaks, quiet)) > 500
    for n in (5, 40):
        grid = roc_grid(peaks, quiet, n_points=n)
        assert len(grid.tau) <= n
        assert TAU_INVESTIGATE in grid.tau and op.tau in grid.tau
        assert set(grid.tau) <= set(candidates(peaks, quiet))
        assert grid.pd[list(grid.tau).index(op.tau)] == pytest.approx(op.pd)
    floor = roc_grid(np.array([[8.0]]), [(np.array([9.0, 9.0, 9.0]), 0.5)])
    assert floor.tau[-1] == np.nextafter(9.0, np.inf)  # the operating point even when off-grid


def test_vectorised_bootstrap_equals_a_naive_loop() -> None:
    peaks, quiet = _random_case(3, n_seeds=30, n_tactics=3, n_quiet=5)
    row_w, quiet_w, per_tactic = roc._replicates(peaks, quiet, 60, 7, 1.0, TAU_INVESTIGATE)
    assert row_w.sum(axis=1).tolist() == [30] * 60 and quiet_w.sum(axis=1).tolist() == [5] * 60
    for b in range(60):
        rows = np.repeat(np.arange(30), row_w[b])  # the same replicate, spelled out
        runs = [quiet[j] for j in np.repeat(np.arange(5), quiet_w[b])]
        naive = operating_point(peaks[rows], runs)
        mine, _ = pd_by_tactic(peaks[rows], naive.tau)
        expected = np.zeros(3) if naive.flag == FLAG_FAR_FLOOR else mine
        assert per_tactic[b] == pytest.approx(expected)


def test_bootstrap_is_reproducible_changes_with_seed_and_contains_the_estimate() -> None:
    peaks, quiet = _random_case(4, n_seeds=150, n_tactics=4, n_quiet=20)
    op = operating_point(peaks, quiet)
    a, b, c = (
        bootstrap(peaks, quiet, seed=1),
        bootstrap(peaks, quiet, seed=1),
        bootstrap(peaks, quiet, seed=2),
    )
    assert a.pd_ci == b.pd_ci and np.array_equal(a.grid_pd_ci, b.grid_pd_ci)
    assert a.pd_ci != c.pd_ci
    assert a.pd_ci[0] <= op.pd <= a.pd_ci[1]
    per_tactic, worst = pd_by_tactic(peaks, op.tau)
    assert a.worst_index == worst and a.worst_pd_ci[0] <= per_tactic[worst] <= a.worst_pd_ci[1]
    grid = roc_grid(peaks, quiet)
    assert np.array_equal(a.grid_tau, grid.tau) and a.grid_pd_ci.shape == (len(grid.tau), 2)
    assert np.all(a.grid_pd_ci[:, 0] <= grid.pd + 1e-12) and np.all(
        grid.pd <= a.grid_pd_ci[:, 1] + 1e-12
    )


def test_clustering_makes_the_interval_wider_than_independent_cells_would() -> None:
    # Every tactic in a row is identical, so the rows are the only source of variation: 40 seeds
    # carry 40 pieces of information however many tactic columns repeat them.
    n_seeds, n_tactics = 40, 10
    rng = np.random.default_rng(5)
    row = np.where(rng.random(n_seeds) < 0.5, 10.0, NEVER_SEEN)
    peaks = np.repeat(row[:, None], n_tactics, axis=1)
    quiet = [(np.array([]), 1.0)] * 4
    op = operating_point(peaks, quiet)
    boot = bootstrap(peaks, quiet, n_boot=2000, seed=0)
    k = int((peaks >= op.tau).sum())
    wrong = wilson_interval(k, n_seeds * n_tactics)  # pretends 400 independent cells
    right = wilson_interval(k // n_tactics, n_seeds)  # 40 independent rows
    boot_width, wrong_width = boot.pd_ci[1] - boot.pd_ci[0], wrong[1] - wrong[0]
    assert boot_width > 2.5 * wrong_width  # about sqrt(10) = 3.2 times wider
    assert boot_width == pytest.approx(right[1] - right[0], rel=0.15)


def test_wilson_interval() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 20)[0] == 0.0 and 0.0 < wilson_interval(0, 20)[1] < 0.2
    assert wilson_interval(20, 20)[1] == 1.0
    with pytest.raises(ValueError, match="k="):
        wilson_interval(5, 4)


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        operating_point(np.array([1.0, 2.0]), [(np.array([]), 1.0)])
    with pytest.raises(ValueError, match="positive number of hours"):
        operating_point(np.array([[1.0]]), [])
    with pytest.raises(ValueError, match="positive number of hours"):
        far_curve([(np.array([3.0]), 0.0)], np.array([1.0]))
