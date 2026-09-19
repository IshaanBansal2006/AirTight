"""The shared demo scene (decisions 017/018): what the simulator renders must be
what the sensing harness and the prior map describe."""

from __future__ import annotations

import numpy as np

from airtight.swarm.scene import SIGN_CENTER_HEIGHT, SIGN_EXTENTS, VEHICLE_EXTENTS, demo_scene


def test_signs_have_standard_faces_at_standard_height() -> None:
    scene = demo_scene()
    for s in scene.signs:
        np.testing.assert_array_equal(s.extent, SIGN_EXTENTS[s.sign_type])
        assert s.position[2] == SIGN_CENTER_HEIGHT
    assert {s.sign_type for s in scene.signs} == {"stop", "speed_limit"}


def test_vehicles_vary_in_size_and_colour_and_sit_on_the_ground() -> None:
    scene = demo_scene()
    assert len({v.body for v in scene.vehicles}) > 1
    assert len({v.color for v in scene.vehicles}) == len(scene.vehicles)
    for v in scene.vehicles:
        assert abs(v.position[2] - VEHICLE_EXTENTS[v.body][2] / 2.0) < 1e-12


def test_landmarks_stay_off_the_target_lane() -> None:
    """A parked vehicle in the target's lane would be hit; a sign in it, driven through."""
    scene = demo_scene()
    lane = scene.target_extent[1] / 2.0
    for v in scene.vehicles:
        assert abs(v.position[1]) - v.extent[1] / 2.0 > lane
    for s in scene.signs:
        assert abs(s.position[1]) > lane


def test_ids_are_unique_and_lookup_works() -> None:
    scene = demo_scene()
    ids = [s.sign_id for s in scene.signs] + [v.vehicle_id for v in scene.vehicles]
    assert len(ids) == len(set(ids))
    assert scene.sign("s2").sign_type == "stop"
