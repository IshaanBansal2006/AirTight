"""End-to-end SLAMMOT, simulator-free (decisions 017-019): two drones fly the
road on a synthesised IMU, localise against prior-mapped signs, map the parked
vehicles, and track the moving car through consider-corrected cameras — with
the tracker never seeing a true drone pose."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from airtight.swarm.edge import rotation
from airtight.swarm.edge.config import EgoConfig
from airtight.swarm.edge.pipeline import PlatformInit, SlammotPipeline
from airtight.swarm.scene import demo_scene
from tests.swarm.road_harness import RoadFlight, road_waypoints


def test_two_drone_slammot_end_to_end() -> None:
    scene = replace(demo_scene(), target_start=(10.0, 0.0, 0.75))
    flight = RoadFlight(seed=11, drone_ids=["d0", "d1"], scene=scene)
    fleet = {
        did: PlatformInit(np.asarray(d.position), np.asarray(d.velocity), np.asarray(d.orientation))
        for did, d in flight.world.drones.items()
    }
    pipe = SlammotPipeline(flight.cfg, EgoConfig(imu=flight.imu_cfg), scene, fleet)

    pose_err = {did: [] for did in fleet}
    track_err = []
    claimed_total = 0
    waypoints = {"d0": road_waypoints(y=3.0), "d1": road_waypoints(y=-3.0)}
    for frame in flight.frames(waypoints, duration_s=20.0):
        for did, samples in frame.imu.items():
            for s in samples:
                pipe.imu(did, s)
        res = pipe.cycle(frame.t, frame.detections)
        claimed_total += res.claimed
        for did, ego in pipe.egos.items():
            truth = frame.truth[did]
            pose_err[did].append(float(np.linalg.norm(ego.p - np.asarray(truth.position))))
        if res.tracks:
            best = min(
                np.linalg.norm(tr.state.x[0:3] - frame.target.state[0:3]) for tr in res.tracks
            )
            track_err.append(float(best))

    # localisation: both drones stay within half a metre on average, no repairs
    for did in fleet:
        assert np.mean(pose_err[did]) < 0.5, (did, np.mean(pose_err[did]))
        assert pose_err[did][-1] < 0.8, (did, pose_err[did][-1])
        assert pipe.egos[did].repairs == 0
    # tracking through estimated poses: the car is confirmed and followed
    assert len(track_err) > 100
    assert np.mean(track_err[-50:]) < 1.0, np.mean(track_err[-50:])
    assert pipe.tracker.consider_repairs == 0
    # the moving car was CLAIMED by the tracker, never mapped as a landmark
    assert claimed_total > 50
    for did, ego in pipe.egos.items():
        assert all(lm.color != scene.target_color for lm in ego.landmarks), did
    # ...while parked vehicles were mapped by at least one drone
    mapped = {lm.color for ego in pipe.egos.values() for lm in ego.landmarks}
    assert len(mapped) >= 2, mapped
    # the wire: valid pose messages with a factor the operating picture can slice
    for msg in res.poses.poses:
        L = np.asarray(msg.pose_sqrt_cov).reshape(6, 6)
        assert np.allclose(np.triu(L, 1), 0.0) and np.all(np.diag(L) > 0.0)
        assert abs(np.linalg.norm(msg.orientation) - 1.0) < 1e-9
    # the two drones each carry their own consider block on the shared track
    tr = res.tracks[0]
    assert set(tr.consider_xc) >= {"d0", "d1"}
    # heading, not just position, is estimated (a 1 deg heading error is 1.75 m at 100 m)
    for did, ego in pipe.egos.items():
        dth = np.linalg.norm(rotation.boxminus(ego.q, np.asarray(frame.truth[did].orientation)))
        assert dth < np.deg2rad(3.0), (did, np.rad2deg(dth))
