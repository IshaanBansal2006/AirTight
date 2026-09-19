from __future__ import annotations

import math

from airtight.contracts import XY
from airtight.redteam.geometry import (
    angle_diff_deg,
    distance_to_boundary,
    path_inside_polygon,
    path_length,
    point_in_polygon,
    progress_along,
)

SQUARE = [XY(x=0, y=0), XY(x=10, y=0), XY(x=10, y=10), XY(x=0, y=10)]
CONCAVE = [XY(x=0, y=0), XY(x=10, y=0), XY(x=10, y=10), XY(x=5, y=5), XY(x=0, y=10)]


def test_point_in_square() -> None:
    assert point_in_polygon(XY(x=5, y=5), SQUARE)
    assert not point_in_polygon(XY(x=15, y=5), SQUARE)
    assert not point_in_polygon(XY(x=5, y=-1), SQUARE)


def test_point_in_concave_notch() -> None:
    assert not point_in_polygon(XY(x=5, y=8), CONCAVE)
    assert point_in_polygon(XY(x=2, y=2), CONCAVE)


def test_distance_to_boundary() -> None:
    assert math.isclose(distance_to_boundary(XY(x=5, y=5), SQUARE), 5.0)
    assert math.isclose(distance_to_boundary(XY(x=10, y=3), SQUARE), 0.0)


def test_path_inside_polygon_catches_leg_that_exits() -> None:
    assert path_inside_polygon([XY(x=1, y=1), XY(x=9, y=9)], SQUARE)
    assert not path_inside_polygon([XY(x=1, y=9), XY(x=9, y=9)], CONCAVE)


def test_path_length_and_progress() -> None:
    assert math.isclose(path_length([XY(x=0, y=0), XY(x=3, y=4), XY(x=3, y=0)]), 9.0)
    assert math.isclose(progress_along(XY(x=5, y=7), XY(x=0, y=0), XY(x=10, y=0)), 0.5)


def test_angle_diff_wraps() -> None:
    assert math.isclose(angle_diff_deg(179, -179), 2.0)
    assert math.isclose(angle_diff_deg(90, 0), 90.0)
