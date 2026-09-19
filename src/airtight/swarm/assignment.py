"""Linear sum assignment — hand-written Hungarian / Jonker-Volgenant.

Layer-neutral, like `schemas.py`: L1 uses it inside Murty's k-best joint
association, and the evaluation harness uses it to match tracks to ground truth.
Neither owns it.

Written rather than pulled from scipy, deliberately. This project is numpy +
pydantic and nothing else, and adding ~40 MB of scipy for one function is a poor
trade when the function is 60 lines and is exactly the kind of algorithm the
project exists to be able to defend from memory.

THE ALGORITHM (shortest augmenting path with potentials, O(n^3)):

The assignment problem is a min-cost perfect matching on a bipartite graph. The
naive view — try every permutation — is O(n!). The Hungarian insight is that the
optimum is unchanged if you subtract a constant from any row or column, so you
can maintain "potentials" u[i], v[j] with the invariant

    reduced_cost(i, j) = cost[i][j] - u[i] - v[j] >= 0                (*)

and grow the matching only along edges where the reduced cost is ZERO. By
complementary slackness, a perfect matching using only zero-reduced-cost edges
is optimal for the ORIGINAL costs. Each of the n outer iterations adds one row
to the matching by finding a shortest augmenting path in the reduced graph
(Dijkstra-like, hence the `minv` distance array), then shifting potentials by
the path length `delta` so that at least one new edge becomes tight — which is
what keeps invariant (*) while making progress.

This is the standard "e-maxx" formulation: 1-indexed internally with a virtual
row 0 as the path sentinel, which is why the arrays are sized n+1 / m+1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Cost to use for a forbidden pairing. Large and FINITE on purpose: np.inf
# breaks the potential update (inf - inf), so the algorithm rejects non-finite
# costs outright rather than silently producing garbage.
FORBIDDEN = 1e12


def linear_sum_assignment(
    cost: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Minimum-cost assignment of rows to columns.

    Returns `(row_idx, col_idx)` such that `cost[row_idx, col_idx].sum()` is
    minimal and each index appears at most once. Rectangular input is fine:
    `min(n, m)` pairs are returned, and the larger side is left partly
    unassigned. Row order in the output is ascending.

    Forbidden pairings: pass `FORBIDDEN` (not `inf`) as the cost.

    Raises:
        ValueError: if `cost` is not 2-D or contains non-finite entries — both
            are caller bugs that would otherwise surface as a wrong answer
            rather than an error.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {cost.shape}")
    if cost.size and not np.isfinite(cost).all():
        raise ValueError(
            "cost contains inf or nan; use assignment.FORBIDDEN (a large finite "
            "value) to forbid a pairing — inf breaks the potential update"
        )
    if cost.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # The algorithm requires rows <= cols; transpose and swap back at the end.
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    parent = np.zeros(m + 1, dtype=np.int64)  # parent[j] = row matched to column j
    way = np.zeros(m + 1, dtype=np.int64)  # predecessor column on the augmenting path

    for i in range(1, n + 1):
        parent[0] = i  # virtual column 0 holds the row we are inserting
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:  # grow the shortest augmenting path one column at a time
            used[j0] = True
            i0 = parent[j0]
            delta, j1 = np.inf, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            # Shift potentials by delta: tight edges stay tight, and the edge to
            # j1 becomes tight. This is what preserves reduced_cost >= 0.
            for j in range(m + 1):
                if used[j]:
                    u[parent[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if parent[j0] == 0:  # reached a free column — the path is complete
                break

        while j0:  # walk the path back, flipping matched/unmatched
            j1 = way[j0]
            parent[j0] = parent[j1]
            j0 = j1

    cols = np.zeros(n, dtype=np.int64)
    for j in range(1, m + 1):
        if parent[j]:
            cols[parent[j] - 1] = j - 1
    rows = np.arange(n, dtype=np.int64)

    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows)
        return rows[order], cols[order]
    return rows, cols


def assignment_cost(cost: NDArray[np.float64], rows: NDArray, cols: NDArray) -> float:
    """Total cost of an assignment — the objective these solvers minimise."""
    return float(np.asarray(cost, dtype=float)[rows, cols].sum())
