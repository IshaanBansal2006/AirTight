from __future__ import annotations

from airtight.contracts.site import XY
from airtight.dimos_lane.person import (
    INTRUDER_RANGE_M,
    default_intruder_xy,
    go2_spawn_xy,
    interpolate_path,
    mujoco_start_pos,
)
from airtight.dimos_lane.site_io import load_example_site


def test_intruder_stands_in_front_of_dock_a() -> None:
    site = load_example_site()
    spawn = go2_spawn_xy(site)
    xy = default_intruder_xy(site)
    assert spawn == site.docks[0].position
    assert abs(xy.x - (spawn.x + INTRUDER_RANGE_M)) < 1e-9
    assert abs(xy.y - spawn.y) < 1e-9
    assert mujoco_start_pos(site) == "10.0,70.0"


def test_scripted_path_is_monotonic() -> None:
    samples = interpolate_path(
        [XY(x=10, y=70), XY(x=16, y=70), XY(x=16, y=64)],
        speed_mps=2.0,
        dt=0.5,
    )
    times = [t for t, _ in samples]
    assert times == sorted(times)
    assert samples[0][1] == XY(x=10, y=70)
    assert samples[-1][1] == XY(x=16, y=64)
    assert samples[-1][0] == 6.0
