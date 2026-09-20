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
WEIGHT_MODES = ("asset", "uniform", "band")


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
    mode: str = "asset",
    r_c: float = 0.0,
) -> Array:
    """Where the patrol should spend its time. Zero outside the fence in every mode.

    "asset":   inside * (base + asset_gain * sum over assets of exp(-distance / scale_m)).
               asset_gain = 0 gives a uniform weight inside the fence.
    "uniform": inside * base.
    "band":    inside * (base + asset_gain where the distance to the nearest asset is >= r_c).
               r_c is the critical ring: a detection inside it is already too late, so the
               band mode spends the extra weight outside it. Inside the ring only base.
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"unknown weight mode {mode!r}; choose one of {WEIGHT_MODES}")
    centers = grid.cell_centers()
    assets = np.asarray(asset_positions, dtype=np.float64).reshape(-1, 2)
    extra: Array = np.zeros(grid.shape, dtype=np.float64)
    if mode == "asset" and len(assets):
        distance = np.linalg.norm(centers[..., None, :] - assets, axis=-1)
        extra = asset_gain * np.exp(-distance / scale_m).sum(axis=-1)
    elif mode == "band" and len(assets):
        nearest = np.linalg.norm(centers[..., None, :] - assets, axis=-1).min(axis=-1)
        extra = asset_gain * (nearest >= r_c).astype(np.float64)
    weight: Array = inside * (base + extra)
    return weight


def polyline_length(points: Array) -> float:
    """Total length of a polyline given as (n, 2). A single point has length 0."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


class Polyline:
    """Segment lengths and cumulative arc length, built once for repeated position queries.

    Intruder and benign objects ask for a position every look. Rebuilding ``diff`` / ``cumsum``
    on each call was the whole cost of those queries; the numbers are identical to a fresh
    ``polyline_position`` on the same vertices.
    """

    __slots__ = ("pts", "seg", "cum", "total")

    def __init__(self, points: Array) -> None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            raise ValueError("polyline needs at least one point")
        self.pts = np.ascontiguousarray(pts)
        if len(pts) == 1:
            self.seg = np.empty(0, dtype=np.float64)
            self.cum = np.array([0.0], dtype=np.float64)
            self.total = 0.0
            return
        self.seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.total = float(self.seg.sum())
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)])

    def position(self, speed: float, t: float) -> tuple[Array, bool]:
        travelled = max(0.0, speed * t)
        if travelled >= self.total - _EPS:
            return self.pts[-1].copy(), True
        i = int(np.searchsorted(self.cum, travelled, side="right")) - 1
        i = min(max(i, 0), len(self.seg) - 1)
        frac = (travelled - self.cum[i]) / self.seg[i] if self.seg[i] > 0 else 0.0
        xy: Array = self.pts[i] + frac * (self.pts[i + 1] - self.pts[i])
        return xy, False


def polyline_position(points: Array, speed: float, t: float) -> tuple[Array, bool]:
    """Position after travelling speed * t along the polyline, clamped to the last point.

    t <= 0 returns the first point. done is True once the whole length has been travelled, so
    a single-point path is done at every t.
    """
    return Polyline(points).position(speed, t)


def in_wedge(origin: Array, heading: float, fov_deg: float, points: Array) -> BoolArray:
    """True where a point lies inside the wedge of fov_deg centred on heading, seen from origin.

    points is (..., 2); the result has shape points.shape[:-1]. fov_deg >= 360 is always True,
    and so is a point at distance 0, whose bearing is undefined. Range is not checked here.
    """
    pts = points if isinstance(points, np.ndarray) else np.asarray(points, dtype=np.float64)
    if fov_deg >= 360.0:
        return np.ones(pts.shape[:-1], dtype=np.bool_)
    if pts.ndim == 1:
        dx = float(pts[0] - origin[0])
        dy = float(pts[1] - origin[1])
        if dx == 0.0 and dy == 0.0:
            return np.bool_(True)
        delta = math.atan2(dy, dx) - heading
        off = (delta + math.pi) % (2.0 * math.pi) - math.pi
        return np.bool_(abs(off) <= math.radians(fov_deg) / 2.0)
    dx = pts[..., 0] - origin[0]
    dy = pts[..., 1] - origin[1]
    # wrap Δbearing to [-π, π] without allocating a complex exponential
    delta = np.arctan2(dy, dx) - heading
    off_axis = np.mod(delta + math.pi, 2.0 * math.pi) - math.pi
    half = math.radians(fov_deg) / 2.0
    inside: BoolArray = (np.abs(off_axis) <= half) | ((dx == 0) & (dy == 0))
    return inside


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
    if not peers:
        return np.ones(centres.shape[:2], dtype=np.bool_)
    own = np.asarray(own_xy, dtype=np.float64)
    d_own = ((centres - own) ** 2).sum(axis=-1)  # squared distance keeps ties exact
    # AND over peers in 2-D: same predicate as a stacked (rows, cols, n_peers) all-reduce,
    # without the 3-D temporary. AND is commutative, so dict order does not matter.
    mask: BoolArray = np.ones(centres.shape[:2], dtype=np.bool_)
    for pid, pxy in peers.items():
        d_p = ((centres - np.asarray(pxy, dtype=np.float64)) ** 2).sum(axis=-1)
        mask &= (d_own < d_p) | ((d_own == d_p) & (own_id < pid))
    return mask
