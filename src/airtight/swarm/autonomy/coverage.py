"""Coverage planning.

Grid-based implementation, deliberately: the area polygon is discretized into
sample points; each point joins the cell of its NEAREST drone seed (a discrete
Voronoi partition); each cell's points are serpentine-ordered into a sweep
route and emitted as one coverage task. Frontier exploration (the unknown-map
upgrade the user plans for hardware) is also grid-based, so `FrontierCoverage`
later swaps in behind the same `CoveragePlanner` protocol reusing this grid
machinery — the 020-style seam, applied to coverage.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Protocol

import numpy as np

from airtight.swarm.autonomy.decomposer import Task

if TYPE_CHECKING:
    from airtight.swarm.autonomy.world_state import WorldState
    from airtight.swarm.schemas import StructuredIntent

_scan_counter = itertools.count()


class CoveragePlanner(Protocol):
    """The 022 seam: VoronoiCoverage today; FrontierCoverage for unknown maps."""

    def plan(self, intent: StructuredIntent, world: WorldState) -> list[Task]: ...


def _points_in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray casting, vectorised over points. Same rule as the previous scalar helper."""
    inside = np.zeros(px.shape, dtype=np.bool_)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < x_cross)
    return inside


class VoronoiCoverage:
    """Discrete Voronoi partition of a polygon + serpentine intra-cell routes."""

    def __init__(self, grid_step: float = 2.0, altitude: float = 2.0) -> None:
        self.grid_step = grid_step
        self.altitude = altitude

    def plan(self, intent: StructuredIntent, world: WorldState) -> list[Task]:
        if not intent.area or len(intent.area) < 3:
            raise ValueError("scan intent requires 'area' polygon (>=3 vertices)")
        drones = world.available_drones()
        if not drones:
            raise ValueError("scan intent with no available drones — nothing to seed cells")

        poly = np.asarray([[float(x), float(y)] for x, y in intent.area])
        xs = np.arange(poly[:, 0].min(), poly[:, 0].max() + 1e-9, self.grid_step)
        ys = np.arange(poly[:, 1].min(), poly[:, 1].max() + 1e-9, self.grid_step)
        # Original nested loop was x-major then y (`for x in xs for y in ys`).
        gx = np.repeat(xs, len(ys))
        gy = np.tile(ys, len(xs))
        inside = _points_in_polygon(gx, gy, poly)
        points = np.column_stack([gx[inside], gy[inside]])
        if len(points) == 0:
            raise ValueError(
                f"grid_step={self.grid_step} produced no sample points inside the "
                f"polygon — decrease grid_step or enlarge the area"
            )

        # discrete Voronoi: each grid point joins its nearest drone seed
        seeds = np.asarray([d.position[:2] for d in drones])
        owner = np.argmin(np.linalg.norm(points[:, None, :] - seeds[None, :, :], axis=2), axis=1)

        tasks: list[Task] = []
        for k in range(len(drones)):
            cell = points[owner == k]
            if len(cell) == 0:
                continue  # seed's cell fell entirely outside the polygon
            tasks.append(
                Task(
                    task_id=f"scan-{next(_scan_counter)}",
                    intent_id=intent.intent_id,
                    task_type="sweep_cell",
                    waypoints=[
                        [float(x), float(y), self.altitude] for x, y in self._serpentine(cell)
                    ],
                    priority=intent.priority,
                )
            )
        return tasks

    def _serpentine(self, cell: np.ndarray) -> list[np.ndarray]:
        """Order a cell's grid points row-by-row, alternating direction —
        a boustrophedon ROUTE inside the Voronoi CELL (routing detail, not the
        partition decision)."""
        order: list[np.ndarray] = []
        for i, y in enumerate(sorted(set(np.round(cell[:, 1], 6)))):
            row = cell[np.isclose(cell[:, 1], y)]
            row = row[np.argsort(row[:, 0])]
            order.extend(row[::-1] if i % 2 else row)
        return order
