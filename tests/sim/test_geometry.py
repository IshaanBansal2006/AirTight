from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from airtight.sim import geometry
from airtight.sim.geometry import (
    Grid,
    inside_mask,
    patrol_weight,
    polyline_length,
    polyline_position,
    voronoi_mask,
)

RECT = np.array([[20.0, 10.0], [50.0, 10.0], [50.0, 30.0], [20.0, 30.0]])
# An L: the 40x40 square from (0, 0) with its top-right 20x20 notch removed.
L_SHAPE = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 20.0], [20.0, 40.0], [0.0, 40.0]])


def test_grid_shape_and_centres_with_offset_origin() -> None:
    grid = Grid(10.0, 20.0, 40.0, 40.0, 5.0)
    assert grid.shape == (4, 6) and grid.n_cells == 24  # rows along y, cols along x
    centers = grid.cell_centers()
    assert centers.shape == (4, 6, 2)
    assert centers[0, 0].tolist() == [12.5, 22.5]
    assert centers[0, 5].tolist() == [37.5, 22.5]  # moving along a row changes x
    assert centers[3, 0].tolist() == [12.5, 37.5]  # moving down the rows changes y


def test_grid_far_edge_rounds_up() -> None:
    grid = Grid(10.0, 20.0, 22.0, 31.0, 5.0)  # 12 m wide, 11 m tall
    assert grid.shape == (3, 3)
    assert Grid(0.0, 0.0, 0.7, 0.3, 0.1).shape == (3, 7)  # no extra cell from float division


def test_grid_centres_are_cached_and_read_only() -> None:
    grid = Grid(0.0, 0.0, 10.0, 10.0, 5.0)
    assert grid.cell_centers() is grid.cell_centers()
    with pytest.raises(ValueError, match="read-only"):
        grid.cell_centers()[0, 0, 0] = 99.0


def test_grid_is_frozen_and_validates() -> None:
    grid = Grid(0.0, 0.0, 10.0, 10.0, 5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.cell_size = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="cell_size"):
        Grid(0.0, 0.0, 10.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="min < max"):
        Grid(10.0, 0.0, 10.0, 10.0, 5.0)


def test_cell_of_inside_far_edge_and_outside() -> None:
    grid = Grid(10.0, 20.0, 40.0, 40.0, 5.0)
    assert grid.cell_of(10.0, 20.0) == (0, 0)
    assert grid.cell_of(16.0, 31.0) == (2, 1)  # (row from y, col from x)
    assert grid.cell_of(40.0, 40.0) == (3, 5)  # far corner belongs to the last cell
    assert grid.cell_of(40.0, 20.0) == (0, 5)
    for x, y in [(9.99, 25.0), (40.01, 25.0), (25.0, 19.99), (25.0, 40.01)]:
        with pytest.raises(ValueError, match="outside"):
            grid.cell_of(x, y)


def test_inside_mask_rectangle() -> None:
    grid = Grid(0.0, 0.0, 60.0, 40.0, 5.0)
    inside = inside_mask(grid, RECT)
    assert inside.shape == grid.shape and inside.dtype == np.bool_
    assert inside[grid.cell_of(35.0, 20.0)]
    assert not inside[grid.cell_of(5.0, 20.0)]
    assert not inside[grid.cell_of(35.0, 37.0)]
    assert int(inside.sum()) == 6 * 4  # a 30 x 20 rectangle of 5 m cells


def test_inside_mask_concave_notch_is_outside() -> None:
    grid = Grid(0.0, 0.0, 40.0, 40.0, 5.0)
    inside = inside_mask(grid, L_SHAPE)
    assert inside[grid.cell_of(10.0, 10.0)]
    assert inside[grid.cell_of(30.0, 10.0)]  # the foot of the L
    assert inside[grid.cell_of(10.0, 30.0)]  # the stem of the L
    assert not inside[grid.cell_of(30.0, 30.0)]  # inside the notch
    assert int(inside.sum()) == 64 - 16
    reversed_winding = inside_mask(grid, L_SHAPE[::-1])
    assert np.array_equal(inside, reversed_winding)


def test_inside_mask_rejects_degenerate_polygon() -> None:
    with pytest.raises(ValueError, match="perimeter"):
        inside_mask(Grid(0.0, 0.0, 10.0, 10.0, 5.0), RECT[:2])


def test_patrol_weight_zero_outside_and_peaks_at_asset() -> None:
    grid = Grid(0.0, 0.0, 60.0, 40.0, 5.0)
    inside = inside_mask(grid, RECT)
    asset = np.array([[41.0, 16.0]])
    weight = patrol_weight(grid, inside, asset)
    assert weight.shape == grid.shape
    assert np.all(weight[~inside] == 0.0)
    assert np.all(weight[inside] > 0.3)
    peak = np.unravel_index(int(np.argmax(weight)), weight.shape)
    assert (int(peak[0]), int(peak[1])) == grid.cell_of(41.0, 16.0)


def test_patrol_weight_is_uniform_inside_when_asset_gain_is_zero() -> None:
    grid = Grid(0.0, 0.0, 60.0, 40.0, 5.0)
    inside = inside_mask(grid, RECT)
    weight = patrol_weight(grid, inside, np.array([[41.0, 16.0]]), asset_gain=0.0)
    assert np.all(weight[inside] == 0.3)
    assert np.all(weight[~inside] == 0.0)


def test_patrol_weight_sums_over_assets() -> None:
    grid = Grid(0.0, 0.0, 60.0, 40.0, 5.0)
    inside = np.ones(grid.shape, dtype=np.bool_)
    a, b = np.array([[10.0, 10.0]]), np.array([[50.0, 30.0]])
    both = patrol_weight(grid, inside, np.vstack([a, b]), base=0.0)
    summed = patrol_weight(grid, inside, a, base=0.0) + patrol_weight(grid, inside, b, base=0.0)
    assert np.allclose(both, summed)


def test_polyline_length() -> None:
    assert polyline_length(np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]])) == 11.0
    assert polyline_length(np.array([[7.0, 7.0]])) == 0.0


def test_polyline_position_before_start_and_halfway() -> None:
    line = np.array([[10.0, 10.0], [20.0, 10.0]])
    for t in (-5.0, 0.0):
        xy, done = polyline_position(line, 2.0, t)
        assert xy.tolist() == [10.0, 10.0] and not done
    xy, done = polyline_position(line, 2.0, 2.5)
    assert xy.tolist() == [15.0, 10.0] and not done


def test_polyline_position_at_and_past_the_end() -> None:
    line = np.array([[10.0, 10.0], [20.0, 10.0]])
    xy, done = polyline_position(line, 2.5, 4.0)
    assert xy.tolist() == [20.0, 10.0] and done
    speed = 0.7  # length / speed * speed is not exactly length in floats
    xy, done = polyline_position(line, speed, polyline_length(line) / speed)
    assert xy.tolist() == [20.0, 10.0] and done
    xy, done = polyline_position(line, 2.5, 400.0)
    assert xy.tolist() == [20.0, 10.0] and done


def test_polyline_position_crosses_a_corner() -> None:
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    xy, done = polyline_position(path, 1.0, 10.0)
    assert xy.tolist() == [10.0, 0.0] and not done  # exactly on the corner
    xy, done = polyline_position(path, 1.0, 14.0)
    assert xy.tolist() == [10.0, 4.0] and not done
    xy, done = polyline_position(path, 1.0, 20.0)
    assert xy.tolist() == [10.0, 10.0] and done


def test_polyline_position_single_point_and_does_not_alias_input() -> None:
    point = np.array([[3.0, 4.0]])
    for t in (-1.0, 0.0, 9.0):
        xy, done = polyline_position(point, 2.0, t)
        assert xy.tolist() == [3.0, 4.0] and done
    xy, _ = polyline_position(point, 2.0, 1.0)
    xy[0] = 99.0
    assert point.tolist() == [[3.0, 4.0]]


def test_voronoi_three_agents_partition_the_grid() -> None:
    grid = Grid(10.0, 20.0, 70.0, 60.0, 5.0)
    centres = grid.cell_centers()
    agents = {0: np.array([20.0, 30.0]), 1: np.array([60.0, 30.0]), 2: np.array([40.0, 55.0])}
    masks = [
        voronoi_mask(centres, xy, {k: v for k, v in agents.items() if k != i}, i)
        for i, xy in agents.items()
    ]
    total = sum(m.astype(int) for m in masks)
    assert np.all(total == 1)  # disjoint, and the union is every cell
    assert all(m.any() for m in masks)
    assert masks[0][grid.cell_of(20.0, 30.0)] and masks[2][grid.cell_of(40.0, 55.0)]


def test_voronoi_exact_ties_go_to_the_lower_id() -> None:
    grid = Grid(0.0, 0.0, 40.0, 40.0, 5.0)
    centres = grid.cell_centers()
    same = np.array([17.0, 23.0])
    low = voronoi_mask(centres, same, {7: same}, 3)
    high = voronoi_mask(centres, same, {3: same}, 7)
    assert low.all() and not high.any()
    # a tie line through cell centres: agents mirrored about x = 22.5
    left = voronoi_mask(centres, np.array([12.5, 20.0]), {1: np.array([32.5, 20.0])}, 0)
    right = voronoi_mask(centres, np.array([32.5, 20.0]), {0: np.array([12.5, 20.0])}, 1)
    assert np.all(left.astype(int) + right.astype(int) == 1)
    assert left[:, 4].all()  # the tied column (x = 22.5) goes to id 0


def test_voronoi_no_peers_is_all_true_and_own_id_cannot_be_a_peer() -> None:
    grid = Grid(0.0, 0.0, 40.0, 40.0, 5.0)
    own = np.array([1.0, 1.0])
    assert voronoi_mask(grid.cell_centers(), own, {}, 0).all()
    with pytest.raises(ValueError, match="own_id"):
        voronoi_mask(grid.cell_centers(), own, {0: own}, 0)


def test_geometry_imports_nothing_from_contracts_or_adapt() -> None:
    source = Path(geometry.__file__).read_text()
    assert "airtight.contracts" not in source and "airtight.sim.adapt" not in source
