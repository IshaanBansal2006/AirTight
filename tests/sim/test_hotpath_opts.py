"""Guards for score-preserving hot-path rewrites: same cells, same poses, same polylines."""

from __future__ import annotations

import math

import numpy as np

from airtight.contracts.site import XY
from airtight.dimos_lane.replay import pose_at_or_before
from airtight.sim.fleet import AgentState, PatrolController
from airtight.sim.geometry import (
    Grid,
    Polyline,
    in_wedge,
    polyline_position,
    voronoi_mask,
)
from airtight.sim.sensing import FixedObserver


def _agent(
    xy: tuple[float, float], radius: float = 22.0, fov: float = 90.0, heading: float = 0.0
) -> AgentState:
    pos = np.array(xy, dtype=np.float64)
    return AgentState("d0", 0, pos, heading, pos.copy(), 8.0, radius, fov)


def test_pose_at_or_before_matches_reverse_scan() -> None:
    track = [(0.0, XY(x=0, y=0)), (1.0, XY(x=1, y=0)), (2.5, XY(x=2, y=0))]
    times = [t for t, _ in track]
    for t in (-1.0, 0.0, 0.5, 1.0, 1.7, 2.5, 9.0):
        naive = next((p for ts, p in reversed(track) if ts <= t), None)
        assert pose_at_or_before(track, t) == naive
        assert pose_at_or_before(track, t, times) == naive
    assert pose_at_or_before([], 0.0) is None


def test_polyline_cache_matches_fresh_position() -> None:
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cached = Polyline(path)
    for t in (-1.0, 0.0, 5.0, 10.0, 14.0, 20.0, 99.0):
        xy_a, done_a = cached.position(1.0, t)
        xy_b, done_b = polyline_position(path, 1.0, t)
        assert done_a is done_b
        assert xy_a.tolist() == xy_b.tolist()
    xy, _ = cached.position(1.0, 4.0)
    xy[0] = 99.0
    assert path[0].tolist() == [0.0, 0.0]


def test_voronoi_vectorized_ties_and_partition() -> None:
    grid = Grid(0.0, 0.0, 300.0, 200.0, 5.0)
    centres = grid.cell_centers()
    a = np.array([40.0, 40.0])
    b = np.array([260.0, 160.0])
    left = voronoi_mask(centres, a, {1: b}, 0)
    right = voronoi_mask(centres, b, {0: a}, 1)
    assert np.all(left.astype(int) + right.astype(int) == 1)
    assert voronoi_mask(centres, a, {}, 0).all()


def test_mark_seen_window_matches_full_grid() -> None:
    """The bbox in mark_seen must mark exactly the cells the full-grid disk+wedge would."""
    grid = Grid(0.0, 0.0, 300.0, 200.0, 5.0)
    weight = np.ones(grid.shape, dtype=np.float64)
    agent = _agent((80.0, 90.0), radius=22.0, fov=90.0, heading=math.pi / 4)
    observer = FixedObserver(
        agent_id=agent.agent_id,
        pos=agent.pos,
        heading=agent.heading,
        fov_deg=agent.fov_deg,
        footprint_radius_m=agent.footprint_radius_m,
        sensor_type="drone_camera",
    )
    ctrl = PatrolController(grid, weight, seed=0, t_start=0.0)
    ctrl.mark_seen([observer], t=12.0)

    centres = grid.cell_centers()
    distance = np.hypot(centres[..., 0] - agent.pos[0], centres[..., 1] - agent.pos[1])
    expected = (distance <= agent.footprint_radius_m) & in_wedge(
        agent.pos, agent.heading, agent.fov_deg, centres
    )
    marked = ctrl.last_seen == 12.0
    assert np.array_equal(marked, expected)


def test_mark_seen_360_fov_is_disk_only() -> None:
    """A 360° camera must mark the disk and skip the wedge; drones use this path."""
    grid = Grid(0.0, 0.0, 300.0, 200.0, 5.0)
    weight = np.ones(grid.shape, dtype=np.float64)
    agent = _agent((80.0, 90.0), radius=22.0, fov=360.0, heading=math.pi / 3)
    ctrl = PatrolController(grid, weight, seed=0, t_start=0.0)
    ctrl.mark_seen([agent], t=5.0)
    centres = grid.cell_centers()
    expected = np.hypot(centres[..., 0] - agent.pos[0], centres[..., 1] - agent.pos[1]) <= 22.0
    assert np.array_equal(ctrl.last_seen == 5.0, expected)


def test_fixed_sensor_mask_is_cached_and_identical() -> None:
    grid = Grid(0.0, 0.0, 80.0, 80.0, 5.0)
    weight = np.ones(grid.shape, dtype=np.float64)
    sensor = FixedObserver(
        agent_id="cam",
        pos=np.array([40.0, 40.0]),
        heading=0.0,
        fov_deg=90.0,
        footprint_radius_m=18.0,
        sensor_type="fixed_cam",
    )
    ctrl = PatrolController(grid, weight, seed=0, t_start=0.0)
    ctrl.mark_seen([sensor], t=1.0)
    first = ctrl.last_seen.copy()
    ctrl.mark_seen([sensor], t=2.0)
    assert id(sensor) in ctrl._static_footprint
    assert np.array_equal(ctrl.last_seen == 2.0, first == 1.0)
