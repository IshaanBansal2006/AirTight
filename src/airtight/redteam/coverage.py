from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from airtight.contracts import XY, SensorCurves, Site
from airtight.redteam.geometry import angle_diff_deg, bearing_deg, points_in_polygon


class CoverageMap(BaseModel):
    """Per-cell coverage score over the site bounds; 0 means nothing watches the cell."""

    cell_m: float
    x0: float
    y0: float
    nx: int
    ny: int
    scores: list[list[float]] = Field(description="scores[iy][ix]")
    source: str

    def center(self, ix: int, iy: int) -> XY:
        return XY(x=self.x0 + (ix + 0.5) * self.cell_m, y=self.y0 + (iy + 0.5) * self.cell_m)

    def cells(self) -> list[tuple[XY, float]]:
        return [
            (self.center(ix, iy), self.scores[iy][ix])
            for iy in range(self.ny)
            for ix in range(self.nx)
        ]

    def low_cells(self, site: Site, max_score: float = 0.0) -> list[XY]:
        """Cell centres inside the perimeter whose score is at or below max_score."""
        scores = np.asarray(self.scores, dtype=np.float64)
        iy, ix = np.nonzero(scores <= max_score)
        if iy.size == 0:
            return []
        xs = self.x0 + (ix + 0.5) * self.cell_m
        ys = self.y0 + (iy + 0.5) * self.cell_m
        inside = points_in_polygon(xs, ys, site.perimeter)
        return [XY(x=float(x), y=float(y)) for x, y, ok in zip(xs, ys, inside, strict=True) if ok]

    def score_at(self, p: XY) -> float:
        ix = int((p.x - self.x0) // self.cell_m)
        iy = int((p.y - self.y0) // self.cell_m)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return self.scores[iy][ix]
        return 0.0


class CoverageSource(Protocol):
    def coverage(self, site: Site, curves: SensorCurves) -> CoverageMap: ...


class GeometryCoverage:
    """Hour-1 coverage: fixed sensors by range and field of view, plus a halo around each dock.

    Replaced by lane B's steady-state snapshot once the sim exists; the interface stays.
    """

    def __init__(self, cell_m: float = 5.0, dock_halo_m: float = 15.0) -> None:
        self.cell_m = cell_m
        self.dock_halo_m = dock_halo_m

    def coverage(self, site: Site, curves: SensorCurves) -> CoverageMap:
        xmin, ymin, xmax, ymax = site.bounds
        nx = max(1, math.ceil((xmax - xmin) / self.cell_m))
        ny = max(1, math.ceil((ymax - ymin) / self.cell_m))
        xs = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * self.cell_m
        ys = ymin + (np.arange(ny, dtype=np.float64) + 0.5) * self.cell_m
        grid_x, grid_y = np.meshgrid(xs, ys)
        scores = np.zeros((ny, nx), dtype=np.float64)
        for s in site.fixed_sensors:
            curve = curves.curves.get(s.sensor_type)
            if curve is None:
                raise KeyError(
                    f"fixed sensor {s.id!r} uses sensor_type {s.sensor_type!r} with no curve; known: {sorted(curves.curves)}"
                )
            dx = grid_x - s.position.x
            dy = grid_y - s.position.y
            in_range = np.hypot(dx, dy) <= curve.max_range_m()
            bearing = np.degrees(np.arctan2(dy, dx))
            delta = (bearing - s.heading_deg + 180.0) % 360.0 - 180.0
            in_fov = np.abs(delta) <= curve.fov_deg / 2.0
            scores += (in_range & in_fov).astype(np.float64)
        halo = self.dock_halo_m
        for d in site.docks:
            scores += (np.hypot(grid_x - d.position.x, grid_y - d.position.y) <= halo).astype(
                np.float64
            )
        return CoverageMap(
            cell_m=self.cell_m,
            x0=xmin,
            y0=ymin,
            nx=nx,
            ny=ny,
            scores=scores.tolist(),
            source="geometry",
        )

    @staticmethod
    def _fixed_sensor_score(c: XY, site: Site, curves: SensorCurves) -> float:
        total = 0.0
        for s in site.fixed_sensors:
            curve = curves.curves.get(s.sensor_type)
            if curve is None:
                raise KeyError(
                    f"fixed sensor {s.id!r} uses sensor_type {s.sensor_type!r} with no curve; known: {sorted(curves.curves)}"
                )
            rng_m = math.hypot(c.x - s.position.x, c.y - s.position.y)
            if rng_m > curve.max_range_m():
                continue
            if angle_diff_deg(bearing_deg(s.position, c), s.heading_deg) <= curve.fov_deg / 2:
                total += 1.0
        return total

    def _dock_score(self, c: XY, site: Site) -> float:
        return sum(
            1.0
            for d in site.docks
            if math.hypot(c.x - d.position.x, c.y - d.position.y) <= self.dock_halo_m
        )
