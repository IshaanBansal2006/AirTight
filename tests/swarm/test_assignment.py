"""Hand-written Hungarian — pinned against brute force."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from airtight.swarm.assignment import (
    FORBIDDEN,
    assignment_cost,
    linear_sum_assignment,
)


def _brute_force(cost: np.ndarray) -> float:
    """Optimal cost by exhaustive permutation — the oracle, valid only for tiny n."""
    n, m = cost.shape
    if n <= m:
        return min(
            sum(cost[i, c] for i, c in enumerate(perm))
            for perm in itertools.permutations(range(m), n)
        )
    return min(
        sum(cost[r, j] for j, r in enumerate(perm)) for perm in itertools.permutations(range(n), m)
    )


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4), (2, 5), (5, 2), (3, 6)])
def test_matches_brute_force(shape: tuple[int, int]) -> None:
    """The only test that really matters: is it optimal, on every shape."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        cost = rng.uniform(0.0, 10.0, size=shape)
        rows, cols = linear_sum_assignment(cost)
        assert len(rows) == min(shape)
        assert len(set(rows.tolist())) == len(rows)  # each row at most once
        assert len(set(cols.tolist())) == len(cols)  # each column at most once
        assert np.isclose(assignment_cost(cost, rows, cols), _brute_force(cost))


def test_negative_costs_are_fine() -> None:
    """Potentials handle negative costs; only non-finite ones are rejected.

    Worth pinning because the classic textbook presentation assumes a
    non-negative matrix, and a reader may 'helpfully' add a shift later.
    """
    rng = np.random.default_rng(1)
    cost = rng.uniform(-5.0, 5.0, size=(4, 4))
    rows, cols = linear_sum_assignment(cost)
    assert np.isclose(assignment_cost(cost, rows, cols), _brute_force(cost))


def test_forbidden_pairs_are_avoided_when_an_alternative_exists() -> None:
    cost = np.array([[FORBIDDEN, 1.0], [1.0, FORBIDDEN]])
    rows, cols = linear_sum_assignment(cost)
    assert cols.tolist() == [1, 0]


def test_rows_are_ascending_when_transposed() -> None:
    """A tall matrix transposes internally; the contract is still row-ascending."""
    rng = np.random.default_rng(2)
    cost = rng.uniform(0.0, 1.0, size=(6, 3))
    rows, _ = linear_sum_assignment(cost)
    assert rows.tolist() == sorted(rows.tolist())


def test_non_finite_costs_raise_with_an_actionable_message() -> None:
    with pytest.raises(ValueError, match="FORBIDDEN"):
        linear_sum_assignment(np.array([[np.inf, 1.0], [1.0, 2.0]]))


def test_empty_input_is_empty_output() -> None:
    rows, cols = linear_sum_assignment(np.empty((0, 3)))
    assert len(rows) == 0 and len(cols) == 0
