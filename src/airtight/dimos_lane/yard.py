"""Build a yard from site.json: MuJoCo occupancy grid or DimSim walls.

Occupancy is written at dimOS's OccupancyGrid.from_path default of 0.05 m/cell
so `DIMOS_MUJOCO_ROOM_FROM_OCCUPANCY` loads the yard at 1:1 metres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from dimos.simulation.dimsim.scene_client import SceneClient
    from numpy.typing import NDArray

    from airtight.contracts.site import XY, Site

OCCUPANCY_RESOLUTION_M = 0.05
WALL_THICKNESS_M = 0.4
MARKER_HALF_M = 0.6
OCCUPIED = np.int8(100)
FREE = np.int8(0)


def _world_to_cell(site: Site, x: float, y: float) -> tuple[int, int]:
    xmin, ymin, _, _ = site.bounds
    col = int((x - xmin) / OCCUPANCY_RESOLUTION_M)
    row = int((y - ymin) / OCCUPANCY_RESOLUTION_M)
    return col, row


def _grid_shape(site: Site) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = site.bounds
    width = max(1, int(np.ceil((xmax - xmin) / OCCUPANCY_RESOLUTION_M)))
    height = max(1, int(np.ceil((ymax - ymin) / OCCUPANCY_RESOLUTION_M)))
    return height, width


def _clip_cell(grid: NDArray[np.int8], col: int, row: int) -> tuple[int, int] | None:
    height, width = grid.shape
    if 0 <= row < height and 0 <= col < width:
        return col, row
    return None


def _stamp_disc(grid: NDArray[np.int8], col: int, row: int, radius_cells: int) -> None:
    height, width = grid.shape
    r2 = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > r2:
                continue
            rr, cc = row + dy, col + dx
            if 0 <= rr < height and 0 <= cc < width:
                grid[rr, cc] = OCCUPIED


def _draw_thick_segment(
    grid: NDArray[np.int8], a: XY, b: XY, site: Site, thickness_m: float
) -> None:
    col0, row0 = _world_to_cell(site, a.x, a.y)
    col1, row1 = _world_to_cell(site, b.x, b.y)
    n = max(abs(col1 - col0), abs(row1 - row0), 1)
    radius = max(1, int(round(thickness_m / OCCUPANCY_RESOLUTION_M / 2)))
    for i in range(n + 1):
        t = i / n
        col = int(round(col0 + t * (col1 - col0)))
        row = int(round(row0 + t * (row1 - row0)))
        hit = _clip_cell(grid, col, row)
        if hit is not None:
            _stamp_disc(grid, hit[0], hit[1], radius)


def perimeter_edges(site: Site) -> list[tuple[XY, XY]]:
    pts = site.perimeter
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def site_to_occupancy(site: Site) -> NDArray[np.int8]:
    """Axis-aligned occupancy: free yard, occupied perimeter, asset and docks."""
    grid = np.full(_grid_shape(site), FREE, dtype=np.int8)
    for a, b in perimeter_edges(site):
        _draw_thick_segment(grid, a, b, site, WALL_THICKNESS_M)
    radius = max(1, int(round(MARKER_HALF_M / OCCUPANCY_RESOLUTION_M)))
    for pt in (site.asset, *(d.position for d in site.docks)):
        col, row = _world_to_cell(site, pt.x, pt.y)
        hit = _clip_cell(grid, col, row)
        if hit is not None:
            _stamp_disc(grid, hit[0], hit[1], radius)
    return grid


def write_occupancy_npy(site: Site, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, site_to_occupancy(site))
    return path


def occupancy_counts(grid: NDArray[np.int8]) -> tuple[int, int]:
    occupied = int(np.sum(grid == OCCUPIED))
    free = int(np.sum(grid == FREE))
    return occupied, free


def apply_dimsim(client: SceneClient, site: Site) -> dict[str, Any]:
    """Place perimeter walls and markers in an empty DimSim scene. Stay on LCM."""
    placed: list[str] = []
    for i, (a, b) in enumerate(perimeter_edges(site)):
        client.add_wall(
            a.x,
            a.y,
            b.x,
            b.y,
            height=2.0,
            thickness=WALL_THICKNESS_M,
            name=f"perimeter_{i}",
        )
        placed.append(f"perimeter_{i}")
    for name, pt, color in (
        ("asset", site.asset, 0xCC2222),
        *((dock.id, dock.position, 0x2266CC) for dock in site.docks),
    ):
        client.add_object(
            "box",
            size=(MARKER_HALF_M * 2, 1.0, MARKER_HALF_M * 2),
            name=name,
            position=(pt.x, 0.5, pt.y),
            color=color,
        )
        placed.append(name)
    return {"placed": placed, "site": site.name}
