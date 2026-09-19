from __future__ import annotations

import math
from collections.abc import Sequence

from airtight.contracts import XY

Bounds = tuple[float, float, float, float]


def point_in_polygon(p: XY, polygon: Sequence[XY]) -> bool:
    """Ray casting: count edges crossed by a horizontal ray from p; odd means inside."""
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        crosses = (a.y > p.y) != (b.y > p.y)
        if crosses:
            x_at_y = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if p.x < x_at_y:
                inside = not inside
    return inside


def distance_to_segment(p: XY, a: XY, b: XY) -> float:
    abx, aby = b.x - a.x, b.y - a.y
    length_sq = abx * abx + aby * aby
    if length_sq == 0.0:
        return math.hypot(p.x - a.x, p.y - a.y)
    t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / length_sq
    t = min(1.0, max(0.0, t))
    return math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby))


def distance_to_boundary(p: XY, polygon: Sequence[XY]) -> float:
    n = len(polygon)
    return min(distance_to_segment(p, polygon[i], polygon[(i + 1) % n]) for i in range(n))


def in_bounds(p: XY, bounds: Bounds) -> bool:
    xmin, ymin, xmax, ymax = bounds
    return xmin <= p.x <= xmax and ymin <= p.y <= ymax


def path_length(points: Sequence[XY]) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(points, points[1:], strict=False))


def sample_segment(a: XY, b: XY, n: int) -> list[XY]:
    """n points strictly between a and b, used to check a leg never leaves the perimeter."""
    return [
        XY(x=a.x + (b.x - a.x) * k / (n + 1), y=a.y + (b.y - a.y) * k / (n + 1))
        for k in range(1, n + 1)
    ]


def path_inside_polygon(
    points: Sequence[XY], polygon: Sequence[XY], samples_per_leg: int = 8, tol: float = 0.0
) -> bool:
    """Every vertex and every sampled interior point lies inside, or within tol of the boundary."""

    def ok(p: XY) -> bool:
        return point_in_polygon(p, polygon) or distance_to_boundary(p, polygon) <= tol

    if not all(ok(p) for p in points):
        return False
    return all(
        ok(q)
        for a, b in zip(points, points[1:], strict=False)
        for q in sample_segment(a, b, samples_per_leg)
    )


def progress_along(p: XY, start: XY, end: XY) -> float:
    """Scalar projection of p onto start->end, normalised to [0, 1] at the endpoints."""
    dx, dy = end.x - start.x, end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return 0.0
    return ((p.x - start.x) * dx + (p.y - start.y) * dy) / length_sq


def bearing_deg(frm: XY, to: XY) -> float:
    return math.degrees(math.atan2(to.y - frm.y, to.x - frm.x))


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)
