"""Pure-numpy geometry for the sim: grid, polygon mask, patrol weight, polylines, Voronoi mask.

Imports nothing from contracts and nothing from adapt. Positions are (x, y) in metres. Grid
arrays are always [row, col], which is [y, x]. Bounds may have a non-zero origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]

_EPS = 1e-9


@dataclass(frozen=True)
class Grid:
    """Square cells over (xmin, ymin, xmax, ymax). Rows run along y, cols along x.

    When the extent is not a whole number of cells the far edge rounds up, so the last row or
    column may reach past xmax or ymax.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    cell_size: float
    rows: int = field(init=False)
    cols: int = field(init=False)
    _centers: Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.cell_size <= 0:
            raise ValueError(f"cell_size must be > 0, got {self.cell_size}")
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError(
                f"grid bounds need min < max, got {(self.xmin, self.ymin, self.xmax, self.ymax)}"
            )
        rows = max(1, math.ceil((self.ymax - self.ymin) / self.cell_size - _EPS))
        cols = max(1, math.ceil((self.xmax - self.xmin) / self.cell_size - _EPS))
        xs = self.xmin + (np.arange(cols, dtype=np.float64) + 0.5) * self.cell_size
        ys = self.ymin + (np.arange(rows, dtype=np.float64) + 0.5) * self.cell_size
        grid_x, grid_y = np.meshgrid(xs, ys)  # both (rows, cols)
        centers = np.stack([grid_x, grid_y], axis=-1)
        centers.setflags(write=False)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "_centers", centers)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def cell_centers(self) -> Array:
        """(rows, cols, 2) of (x, y). Computed once; the array is read-only."""
        return self._centers

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """(row, col) of the cell holding (x, y). Points on the far edge belong to the last cell."""
        if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
            raise ValueError(
                f"point ({x}, {y}) is outside grid bounds "
                f"{(self.xmin, self.ymin, self.xmax, self.ymax)}"
            )
        row = min(int((y - self.ymin) // self.cell_size), self.rows - 1)
        col = min(int((x - self.xmin) // self.cell_size), self.cols - 1)
        return (row, col)


def inside_mask(grid: Grid, perimeter: Array) -> BoolArray:
    """(rows, cols) bool: True where the cell centre is inside the polygon.

    Ray casting to the right, vectorised over cells, one pass per edge. Correct for concave
    polygons. Vertices are (n, 2), in order, not repeated at the end.
    """
    poly = np.asarray(perimeter, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(f"perimeter must have shape (n >= 3, 2), got {poly.shape}")
    centers = grid.cell_centers()
    px, py = centers[..., 0], centers[..., 1]
    inside = np.zeros(grid.shape, dtype=np.bool_)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if y1 == y2:
            continue  # a horizontal edge never straddles the ray
        straddles = (y1 > py) != (y2 > py)
        x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddles & (px < x_cross)
    return inside


def patrol_weight(
    grid: Grid,
    inside: BoolArray,
    asset_positions: Array,
    base: float = 0.3,
    scale_m: float = 25.0,
    asset_gain: float = 1.0,
) -> Array:
    """inside * (base + asset_gain * sum over assets of exp(-distance / scale_m)).

    asset_gain = 0 gives a uniform weight inside the fence.
    """
    centers = grid.cell_centers()
    pull = np.zeros(grid.shape, dtype=np.float64)
    for ax, ay in np.asarray(asset_positions, dtype=np.float64).reshape(-1, 2):
        distance = np.hypot(centers[..., 0] - ax, centers[..., 1] - ay)
        pull += np.exp(-distance / scale_m)
    weight: Array = inside * (base + asset_gain * pull)
    return weight


def polyline_length(points: Array) -> float:
    """Total length of a polyline given as (n, 2). A single point has length 0."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def polyline_position(points: Array, speed: float, t: float) -> tuple[Array, bool]:
    """Position after travelling speed * t along the polyline, clamped to the last point.

    t <= 0 returns the first point. done is True once the whole length has been travelled, so
    a single-point path is done at every t.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("polyline needs at least one point")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    travelled = max(0.0, speed * t)
    if travelled >= total - _EPS:
        return pts[-1].copy(), True
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    i = int(np.searchsorted(cum, travelled, side="right")) - 1
    i = min(max(i, 0), len(seg) - 1)
    frac = (travelled - cum[i]) / seg[i] if seg[i] > 0 else 0.0
    xy: Array = pts[i] + frac * (pts[i + 1] - pts[i])
    return xy, False


def voronoi_mask(
    centres: Array,
    own_xy: Array,
    peers: dict[int, Array],
    own_id: int,
) -> BoolArray:
    """(rows, cols) bool: True where the cell centre is nearer to own_xy than to every peer.

    Exact ties go to the lower id, so two agents never both own a cell. No peers means all True.
    centres is (rows, cols, 2) as returned by Grid.cell_centers().
    """
    if own_id in peers:
        raise ValueError(f"own_id {own_id} also appears in peers {sorted(peers)}")
    own = np.asarray(own_xy, dtype=np.float64)
    d_own = ((centres - own) ** 2).sum(axis=-1)  # squared distance keeps ties exact
    mask = np.ones(centres.shape[:2], dtype=np.bool_)
    for peer_id, peer_xy in peers.items():
        d_peer = ((centres - np.asarray(peer_xy, dtype=np.float64)) ** 2).sum(axis=-1)
        mask &= (d_own < d_peer) | ((d_own == d_peer) & (own_id < peer_id))
    return mask
