"""Drone-mounted camera geometry and landmark detection synthesis (017/018)."""

from __future__ import annotations

import numpy as np

from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import CameraConfig, CameraMount, RadarConfig, TrackerConfig
from airtight.swarm.edge.observation import CameraModel, h_camera, h_camera_box, in_view
from airtight.swarm.edge.sensing import (
    MeasurementSynthesizer,
    PlatformTruth,
    TargetTruth,
    camera_model,
    mount_rotation,
)
from airtight.swarm.scene import demo_scene


def project_point(cam: CameraModel, p_w: np.ndarray) -> tuple[float, float, float]:
    p_c = cam.R_wc @ (p_w - cam.t_w)
    return (cam.fx * p_c[0] / p_c[2] + cam.cx, cam.fy * p_c[1] / p_c[2] + cam.cy, p_c[2])


def test_level_mount_looks_along_body_x_with_image_down_equal_body_down() -> None:
    R_bc = mount_rotation(CameraMount(pitch_deg=0.0))
    np.testing.assert_allclose(R_bc @ [0, 0, 1], [1, 0, 0], atol=1e-12)  # cam z -> body x
    np.testing.assert_allclose(R_bc @ [0, 1, 0], [0, 0, -1], atol=1e-12)  # cam y -> body -z
    np.testing.assert_allclose(R_bc @ [1, 0, 0], [0, -1, 0], atol=1e-12)  # cam x -> body -y


def test_thirty_degree_pitch_centres_a_point_thirty_degrees_below_forward() -> None:
    cfg = CameraConfig(platform="d0", mount=CameraMount(pitch_deg=30.0))
    pos, yaw = np.array([2.0, -1.0, 4.0]), 0.6
    cam = camera_model(pos, rotation.from_yaw(yaw), cfg)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    ahead = pos + 10.0 * fwd + np.array([0.0, 0.0, -10.0 * np.tan(np.deg2rad(30.0))])
    u, v, z = project_point(cam, ahead)
    assert abs(u - cfg.cx) < 1e-6 and abs(v - cfg.cy) < 1e-6 and z > 0
    # a level point straight ahead sits ABOVE the image centre (smaller v)
    u2, v2, _ = project_point(cam, pos + 10.0 * fwd)
    assert abs(u2 - cfg.cx) < 1e-6 and v2 < cfg.cy
    # a point to the platform's LEFT lands on the image's left (smaller u)
    left = pos + 10.0 * fwd + 2.0 * np.array([-np.sin(yaw), np.cos(yaw), 0.0]) - [0, 0, 5.77]
    u3, _, _ = project_point(cam, left)
    assert u3 < cfg.cx


def test_mount_offset_moves_the_camera_in_the_body_frame() -> None:
    cfg = CameraConfig(platform="d0", mount=CameraMount(offset=[0.2, 0.0, -0.1]))
    cam = camera_model(np.zeros(3), rotation.from_yaw(np.pi / 2), cfg)
    np.testing.assert_allclose(cam.t_w, [0.0, 0.2, -0.1], atol=1e-12)


def road_config() -> TrackerConfig:
    return TrackerConfig(
        sensors={
            "cam_d0": CameraConfig(platform="d0"),
            "radar_gs": RadarConfig(position=[-15.0, -10.0, 3.0]),
        }
    )


def test_synthesizer_reports_visible_landmarks_with_their_attributes() -> None:
    scene = demo_scene()
    synth = MeasurementSynthesizer(scene, road_config(), np.random.default_rng(0))
    # hover behind sign s0 (at x=0, y=+4), facing +x from 4 m altitude: sign ahead-left,
    # target 13 m ahead on the road (3 m ahead it would be cut off by the image bottom)
    platforms = {"d0": PlatformTruth(np.array([-8.0, 3.0, 4.0]), rotation.from_yaw(0.0))}
    target = TargetTruth(np.array([5.0, 0.0, 0.75, 2.0, 0.0, 0.0, 4.5, 1.8, 1.5]), 0.0, "silver")
    dets, cams = synth.observe(0.0, platforms, [target])
    by_label: dict[str, list] = {}
    for d in dets:
        by_label.setdefault(d.class_label or "?", []).append(d)
    assert [d.sensor_id for d in by_label["sign"]] and all(
        d.attributes["sign_type"] in ("stop", "speed_limit") for d in by_label["sign"]
    )
    colours = {d.attributes.get("color") for d in by_label["vehicle"] if d.sensor_id == "cam_d0"}
    assert "silver" in colours  # the target
    assert len([d for d in dets if d.sensor_id == "radar_gs"]) == 1
    # every camera detection is the noiseless projection plus noise of the configured size
    for d in by_label["sign"]:
        sign = next(
            s
            for s in scene.signs
            if np.allclose(
                h_camera_box(np.asarray(s.position), s.extent, s.yaw, cams["cam_d0"]),
                d.measurement,
                atol=15.0,
            )
        )
        assert sign.sign_type == d.attributes["sign_type"]


def test_synthesizer_reports_nothing_the_camera_cannot_see() -> None:
    scene = demo_scene()
    synth = MeasurementSynthesizer(scene, road_config(), np.random.default_rng(0))
    # facing -x from the far end: every landmark is behind the camera
    platforms = {"d0": PlatformTruth(np.array([-8.0, 3.0, 4.0]), rotation.from_yaw(np.pi))}
    target = TargetTruth(np.array([-5.0, 0.0, 0.75, 2.0, 0.0, 0.0, 4.5, 1.8, 1.5]), 0.0, "silver")
    dets, _ = synth.observe(0.0, platforms, [target])
    assert all(d.sensor_id == "radar_gs" for d in dets)


def test_explicit_box_projection_matches_state_projection() -> None:
    cam = camera_model(
        np.array([0.0, 0.0, 4.0]), rotation.from_yaw(0.0), CameraConfig(platform="d0")
    )
    x = np.array([12.0, 1.0, 0.75, 2.0, 0.5, 0.0, 4.5, 1.8, 1.5])
    yaw = float(np.arctan2(x[4], x[3]))
    np.testing.assert_allclose(h_camera_box(x[0:3], x[6:9], yaw, cam), h_camera(x, cam))
    assert in_view(x, cam)
