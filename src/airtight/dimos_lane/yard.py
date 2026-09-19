"""Build a yard from site.json: MuJoCo occupancy grid or DimSim walls.

Occupancy is written at dimOS's OccupancyGrid.from_path default of 0.05 m/cell
so `DIMOS_MUJOCO_ROOM_FROM_OCCUPANCY` loads the yard at 1:1 metres.
"""

from __future__ import annotations

from dataclasses import dataclass
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
DOCK_MARKER_HALF_M = 0.25
DOCK_MARKER_OFFSET_M = (0.0, -2.0)
MUJOCO_CROP_RADIUS_M = 18.0
OCCUPIED = np.int8(100)
FREE = np.int8(0)


@dataclass(frozen=True)
class YardBlob:
    name: str
    cls: str
    x: float
    y: float
    half_x: float
    half_y: float
    height: float = 1.0
    color: int = 0x888888


def default_benign_blobs() -> tuple[YardBlob, ...]:
    """Two interior objects: a van on the delivery lane and a cart on the staff path.

    First waypoints of the example routes sit on the fence, so these are inset.
    """
    return (
        YardBlob("delivery_van", "vehicle", 20.0, 22.0, 2.0, 0.9, 1.4, 0x334455),
        YardBlob("staff_cart", "person", 45.0, 50.0, 0.4, 0.4, 0.9, 0x886644),
    )


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


def world_cell(grid: NDArray[np.int8], site: Site, x: float, y: float) -> int | None:
    col, row = _world_to_cell(site, x, y)
    hit = _clip_cell(grid, col, row)
    if hit is None:
        return None
    return int(grid[hit[1], hit[0]])


def is_occupied(grid: NDArray[np.int8], site: Site, x: float, y: float) -> bool:
    value = world_cell(grid, site, x, y)
    return value == int(OCCUPIED)


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


def _stamp_aabb(
    grid: NDArray[np.int8],
    site: Site,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> None:
    c0, r0 = _world_to_cell(site, xmin, ymin)
    c1, r1 = _world_to_cell(site, xmax, ymax)
    if c0 > c1:
        c0, c1 = c1, c0
    if r0 > r1:
        r0, r1 = r1, r0
    height, width = grid.shape
    c0 = max(0, c0)
    r0 = max(0, r0)
    c1 = min(width - 1, c1)
    r1 = min(height - 1, r1)
    if c1 >= c0 and r1 >= r0:
        grid[r0 : r1 + 1, c0 : c1 + 1] = OCCUPIED


def _stamp_blob(grid: NDArray[np.int8], site: Site, blob: YardBlob) -> None:
    _stamp_aabb(
        grid,
        site,
        blob.x - blob.half_x,
        blob.y - blob.half_y,
        blob.x + blob.half_x,
        blob.y + blob.half_y,
    )


def _draw_thick_segment(
    grid: NDArray[np.int8], a: XY, b: XY, site: Site, thickness_m: float
) -> None:
    half = thickness_m / 2.0
    if abs(a.y - b.y) < 1e-9:
        _stamp_aabb(grid, site, min(a.x, b.x), a.y - half, max(a.x, b.x), a.y + half)
        return
    if abs(a.x - b.x) < 1e-9:
        _stamp_aabb(grid, site, a.x - half, min(a.y, b.y), a.x + half, max(a.y, b.y))
        return
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


def coarsen(grid: NDArray[np.int8], factor: int = 4) -> NDArray[np.int8]:
    """Stride-sample occupancy. Occupied cells survive if any sampled cell is occupied."""
    if factor <= 1:
        return grid
    return np.array(grid[::factor, ::factor], dtype=np.int8)


def site_to_occupancy(site: Site, *, benign: tuple[YardBlob, ...] | None = None) -> NDArray[np.int8]:
    """Free yard, occupied perimeter / asset / offset docks / two benign blobs."""
    grid = np.full(_grid_shape(site), FREE, dtype=np.int8)
    for a, b in perimeter_edges(site):
        _draw_thick_segment(grid, a, b, site, WALL_THICKNESS_M)
    _stamp_aabb(
        grid,
        site,
        site.asset.x - MARKER_HALF_M,
        site.asset.y - MARKER_HALF_M,
        site.asset.x + MARKER_HALF_M,
        site.asset.y + MARKER_HALF_M,
    )
    ox, oy = DOCK_MARKER_OFFSET_M
    for dock in site.docks:
        _stamp_aabb(
            grid,
            site,
            dock.position.x + ox - DOCK_MARKER_HALF_M,
            dock.position.y + oy - DOCK_MARKER_HALF_M,
            dock.position.x + ox + DOCK_MARKER_HALF_M,
            dock.position.y + oy + DOCK_MARKER_HALF_M,
        )
    for blob in benign if benign is not None else default_benign_blobs():
        _stamp_blob(grid, site, blob)
    return grid


def crop_to_radius(
    grid: NDArray[np.int8], site: Site, x: float, y: float, radius_m: float
) -> NDArray[np.int8]:
    """Keep occupied cells within `radius_m` of (x, y). Used for the live MuJoCo npy.

    The full 120x80 m yard makes dimOS's depth/lidar pass explode; a local crop
    still contains the near walls, dock marker, and the Go2 camera's intruder.
    """
    xmin, ymin, _, _ = site.bounds
    height, width = grid.shape
    world_x = xmin + (np.arange(width, dtype=np.float64) + 0.5) * OCCUPANCY_RESOLUTION_M
    world_y = ymin + (np.arange(height, dtype=np.float64) + 0.5) * OCCUPANCY_RESOLUTION_M
    xx, yy = np.meshgrid(world_x, world_y)
    out = grid.copy()
    out[(xx - x) ** 2 + (yy - y) ** 2 > radius_m * radius_m] = FREE
    return out


def write_occupancy_npy(site: Site, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, site_to_occupancy(site))
    return path


def write_mujoco_occupancy_npy(
    site: Site, path: Path, *, radius_m: float = MUJOCO_CROP_RADIUS_M
) -> Path:
    from airtight.dimos_lane.person import go2_spawn_xy

    spawn = go2_spawn_xy(site)
    cropped = crop_to_radius(site_to_occupancy(site), site, spawn.x, spawn.y, radius_m)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, cropped)
    return path


def occupancy_counts(grid: NDArray[np.int8]) -> tuple[int, int]:
    occupied = int(np.sum(grid == OCCUPIED))
    free = int(np.sum(grid == FREE))
    return occupied, free


def apply_dimsim(
    client: SceneClient,
    site: Site,
    *,
    benign: tuple[YardBlob, ...] | None = None,
) -> dict[str, Any]:
    """Place perimeter walls, markers, and benign objects in an empty DimSim scene."""
    placed: list[str] = []
    blobs = benign if benign is not None else default_benign_blobs()
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
    for blob in blobs:
        client.add_object(
            "box",
            size=(blob.half_x * 2, blob.height, blob.half_y * 2),
            name=blob.name,
            position=(blob.x, blob.height / 2.0, blob.y),
            color=blob.color,
        )
        placed.append(blob.name)
    return {"placed": placed, "site": site.name, "benign": [b.name for b in blobs]}
