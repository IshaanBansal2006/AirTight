"""Camera visibility: when h_camera is a valid model of a detection."""

from __future__ import annotations

import numpy as np

from airtight.swarm.edge.observation import CameraModel, h_camera, in_view

WIDTH, HEIGHT = 960, 600


def camera_at(cam_pos: np.ndarray, look_at: np.ndarray) -> CameraModel:
    z_c = (look_at - cam_pos) / np.linalg.norm(look_at - cam_pos)
    x_c = np.cross(z_c, [0.0, 0.0, 1.0])
    x_c /= np.linalg.norm(x_c)
    return CameraModel(
        fx=480.0,
        fy=480.0,
        cx=480.0,
        cy=300.0,
        width=WIDTH,
        height=HEIGHT,
        R_wc=np.stack([x_c, np.cross(z_c, x_c), z_c]),
        t_w=cam_pos,
    )


def box(center: list[float], extent: float = 1.0) -> np.ndarray:
    return np.array([*center, 0.0, 0.0, 0.0, extent, extent, extent])


FORWARD = camera_at(np.zeros(3), np.array([10.0, 0.0, 0.0]))


def test_box_ahead_is_in_view() -> None:
    assert in_view(box([10.0, 0.0, 0.0]), FORWARD)


def test_box_behind_camera_is_not_in_view() -> None:
    assert not in_view(box([-10.0, 0.0, 0.0]), FORWARD)


def test_box_straddling_camera_plane_is_not_in_view() -> None:
    assert not in_view(box([0.2, 0.0, 0.0]), FORWARD)


def test_box_outside_image_is_not_in_view() -> None:
    assert not in_view(box([10.0, 50.0, 0.0]), FORWARD)


def test_box_truncated_at_image_edge_is_not_in_view() -> None:
    x = box([10.0, 9.8, 0.0])
    u_center = h_camera(x, FORWARD)[0]
    assert 0.0 <= u_center <= WIDTH  # the centre is in the image...
    assert not in_view(x, FORWARD)  # ...but the box is cut off at the left edge


def test_in_view_implies_predicted_box_inside_image() -> None:
    rng = np.random.default_rng(0)
    visible = 0
    for _ in range(500):
        x = box(
            rng.uniform([-20.0, -20.0, -10.0], [40.0, 20.0, 10.0]).tolist(),
            extent=float(rng.uniform(0.2, 4.0)),
        )
        if not in_view(x, FORWARD):
            continue
        visible += 1
        u, v, w, h = h_camera(x, FORWARD)
        assert u - w / 2 >= 0.0 and u + w / 2 <= WIDTH
        assert v - h / 2 >= 0.0 and v + h / 2 <= HEIGHT
    assert visible > 0
