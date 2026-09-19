from __future__ import annotations

from importlib import resources
from pathlib import Path

from airtight.contracts.site import XY
from airtight.dimos_lane.clips import miss_log_from_catch, write_demo_clips
from airtight.dimos_lane.person import interpolate_path
from airtight.dimos_lane.replay import plan_replay
from airtight.dimos_lane.site_io import load_example_site
from airtight.dimos_lane.yard import occupancy_counts, perimeter_edges, site_to_occupancy

EXAMPLE_LOG = Path(str(resources.files("airtight.contracts.examples").joinpath("episode.jsonl")))


def test_yard_occupancy_has_walls_and_open_interior() -> None:
    site = load_example_site()
    grid = site_to_occupancy(site)
    occupied, free = occupancy_counts(grid)
    assert occupied > 0 and free > occupied
    assert len(perimeter_edges(site)) == len(site.perimeter)
    # a cell near the asset should be occupied (marker), centre-north of yard is free-ish
    assert grid.shape[0] == 1600 and grid.shape[1] == 2400


def test_person_path_reaches_last_waypoint() -> None:
    samples = interpolate_path(
        [XY(x=0, y=0), XY(x=10, y=0)],
        speed_mps=2.0,
        dt=0.5,
    )
    assert samples[0][1] == XY(x=0, y=0)
    assert abs(samples[-1][1].x - 10) < 1e-6
    assert samples[-1][0] == 5.0


def test_replay_parses_example_episode() -> None:
    plan = plan_replay(EXAMPLE_LOG)
    assert plan.timely_detected is True
    assert plan.intruder[0][1].x == 60
    assert plan.dispatch_at == 14.0
    assert plan.dispatch_result is not None and "winner=" in plan.dispatch_result


def test_clips_miss_and_catch(tmp_path: Path) -> None:
    miss_html, catch_html = write_demo_clips(EXAMPLE_LOG, tmp_path)
    assert "timely_detected=False" in miss_html.read_text()
    assert "timely_detected=True" in catch_html.read_text()
    miss_log = miss_log_from_catch(EXAMPLE_LOG, tmp_path / "again.jsonl")
    miss_plan = plan_replay(miss_log)
    assert miss_plan.timely_detected is False
    assert miss_plan.t_alarm is None
