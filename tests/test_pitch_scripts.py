from __future__ import annotations

import json
import re
import runpy
import sys
from pathlib import Path

import pytest

PITCH = Path(__file__).parents[1] / "pitch"


@pytest.fixture(autouse=True)
def _pitch_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(PITCH))


def test_make_charts_renders_from_example_report(tmp_path: Path) -> None:
    sys.argv = [
        "make_charts.py",
        "--report",
        str(tmp_path / "missing.json"),
        "--tactics-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "charts"),
    ]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "make_charts.py"), run_name="__main__")
    assert exc.value.code == 0
    names = {p.name for p in (tmp_path / "charts").iterdir()}
    assert {
        "cost_vs_detection.png",
        "roc.png",
        "vulnerability_map.png",
        "before_after.png",
        "numbers.json",
    } <= names
    numbers = json.loads((tmp_path / "charts" / "numbers.json").read_text())
    assert numbers["before_after"]["fixed"] == "3drone_go2_stagger" and numbers["frontier"]


def test_make_token_chart_without_ledger(tmp_path: Path) -> None:
    sys.argv = [
        "make_token_chart.py",
        "--ledger",
        str(tmp_path / "none.jsonl"),
        "--out",
        str(tmp_path / "charts"),
    ]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "make_token_chart.py"), run_name="__main__")
    assert exc.value.code == 0
    numbers = json.loads((tmp_path / "charts" / "token_numbers.json").read_text())
    assert (
        numbers["comparison"]["ratio"] == pytest.approx(100.0)
        and "price table" in numbers["comparison"]["basis"]
    )


def test_build_deck_from_rendered_numbers(tmp_path: Path) -> None:
    charts = tmp_path / "charts"
    sys.argv = [
        "make_charts.py",
        "--report",
        str(tmp_path / "missing.json"),
        "--tactics-dir",
        str(tmp_path),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_charts.py"), run_name="__main__")
    sys.argv = [
        "make_token_chart.py",
        "--ledger",
        str(tmp_path / "none.jsonl"),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_token_chart.py"), run_name="__main__")
    sys.argv = ["build_deck.py", "--charts", str(charts), "--out", str(tmp_path / "deck.md")]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "build_deck.py"), run_name="__main__")
    assert exc.value.code == 0
    deck = (tmp_path / "deck.md").read_text()
    assert deck.count("\n---\n") >= 8 and "EXAMPLE DATA" in deck and "cost_vs_detection.png" in deck


def test_render_replay_from_a_v0_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
    from airtight.sim.runner import run_episode

    monkeypatch.setenv("AIRTIGHT_ENGINE", "v0")
    scen = Path(__file__).parents[1] / "scenarios" / "logistics_yard"
    site = Site.model_validate_json((scen / "site.json").read_text())
    curves = SensorCurves.model_validate_json((scen / "sensor_curve.json").read_text())
    fleet = FleetConfig.model_validate_json(
        (scen / "fleets" / "d2_go2_guard_sync.json").read_text()
    )
    tactic = Tactic(
        id="walk",
        family="charging_window",
        entry_id="main_gate",
        phase=0.1,
        speed_mps=1.5,
        waypoints=[site.asset],
    )
    res = run_episode(site, fleet, tactic, curves, 5, tmp_path / "logs")
    sys.argv = [
        "render_replay.py",
        str(res.log_path),
        "--site",
        str(scen / "site.json"),
        "--out",
        str(tmp_path / "clip.mp4"),
        "--fps",
        "4",
        "--speed",
        "20",
    ]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "render_replay.py"), run_name="__main__")
    assert exc.value.code == 0 and (tmp_path / "clip.mp4").stat().st_size > 1000


def test_fill_writeup_from_rendered_numbers(tmp_path: Path) -> None:
    charts = tmp_path / "charts"
    sys.argv = [
        "make_charts.py",
        "--report",
        str(tmp_path / "missing.json"),
        "--tactics-dir",
        str(tmp_path),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_charts.py"), run_name="__main__")
    sys.argv = [
        "make_token_chart.py",
        "--ledger",
        str(tmp_path / "none.jsonl"),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_token_chart.py"), run_name="__main__")
    sys.argv = ["fill_writeup.py", "--charts", str(charts), "--out", str(tmp_path / "w.md")]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "fill_writeup.py"), run_name="__main__")
    assert exc.value.code == 0
    text = (tmp_path / "w.md").read_text()
    assert "EXAMPLE DATA" in text and "[pd_baseline]" not in text and "[ratio]" not in text


def test_deck_accepts_lane_a_clip_sidecars(tmp_path: Path) -> None:
    sys.path.insert(0, str(PITCH))
    from build_deck import load_clip_facts

    (tmp_path / "miss.json").write_text(
        json.dumps(
            {
                "title": "miss",
                "seed": 7,
                "tactic_id": "t1",
                "timely_detected": False,
                "t_alarm": None,
                "fleet": "base",
            }
        )
    )
    (tmp_path / "catch.json").write_text(
        json.dumps(
            {
                "title": "catch",
                "seed": 7,
                "tactic_id": "t1",
                "timely_detected": True,
                "t_alarm": 14.0,
                "t_cdp": 30.0,
                "fleet": "fixed",
            }
        )
    )
    facts = load_clip_facts(tmp_path)
    assert (
        facts
        and facts["seed"] == 7
        and facts["catch_t_alarm"] == 14.0
        and facts["fixed"] == "fixed"
    )
    assert load_clip_facts(tmp_path / "nowhere") is None


def test_build_app_from_example_report_and_v0_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
    from airtight.sim.runner import run_episode

    monkeypatch.setenv("AIRTIGHT_ENGINE", "v0")
    scen = Path(__file__).parents[1] / "scenarios" / "logistics_yard"
    site = Site.model_validate_json((scen / "site.json").read_text())
    curves = SensorCurves.model_validate_json((scen / "sensor_curve.json").read_text())
    tactic = Tactic(
        id="walk",
        family="charging_window",
        entry_id="main_gate",
        phase=0.1,
        speed_mps=1.5,
        waypoints=[site.asset],
    )
    names = ("d2_go2_guard_sync", "d3_go2_guard_stagger")
    for kind, name in zip(("miss", "catch"), names, strict=True):
        fleet = FleetConfig.model_validate_json((scen / "fleets" / f"{name}.json").read_text())
        run_episode(site, fleet, tactic, curves, 5, tmp_path / "clip_logs" / kind)
    clips = tmp_path / "clips_c.json"
    clips.write_text(
        json.dumps(
            {
                "tactic_id": "walk",
                "family": "charging_window",
                "entry": "main_gate",
                "phase": 0.1,
                "seed": 5,
                "baseline": names[0],
                "fixed": names[1],
                "miss_t_alarm": None,
                "catch_t_alarm": 20.0,
                "t_cdp": 30.0,
            }
        )
    )
    charts = tmp_path / "charts"
    sys.argv = [
        "make_charts.py",
        "--report",
        str(tmp_path / "missing.json"),
        "--tactics-dir",
        str(tmp_path),
        "--out",
        str(charts),
    ]
    with pytest.raises(SystemExit):
        runpy.run_path(str(PITCH / "make_charts.py"), run_name="__main__")
    from importlib import resources

    example = resources.files("airtight.contracts.examples") / "report.json"
    sys.argv = [
        "build_app.py",
        "--report",
        str(example),
        "--charts",
        str(charts),
        "--tactics-dir",
        str(tmp_path),
        "--clips",
        str(clips),
        "--clip-logs",
        str(tmp_path / "clip_logs"),
        "--minmax-dir",
        str(tmp_path / "no_minmax"),
        "--out",
        str(tmp_path / "app.html"),
    ]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PITCH / "build_app.py"), run_name="__main__")
    assert exc.value.code == 0
    html = (tmp_path / "app.html").read_text()
    assert "__DATA__" not in html and '"episodes"' in html and "three.min.js" in html
    assert re.search(r'"rounds":\s*\[\]', html) and '"responder"' in html
    assert not re.search(r"__[A-Z][A-Z_]*__", html)
    data = json.loads(
        re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
        .group(1)
        .replace("<\\/", "</")
    )
    assert set(data["ramp"]) == {"light", "dark"} and len(html) < 400_000
    try:
        import dimos.mapping.voxels.impl.packed  # noqa: F401
    except ImportError:
        assert data["episodes"]["catch"]["view"] is None
    else:
        for kind in ("miss", "catch"):
            view = data["episodes"][kind]["view"]
            assert view["n"] == len(view["pm"]["touch"]) and view["path"][0] == [100.0, 115.0]


def test_column_carving_clears_a_moved_object_only_when_its_column_is_reswept() -> None:
    pytest.importorskip("dimos.mapping.voxels.impl.packed")
    from perception_map import ColumnMap, Sensed

    cmap = ColumnMap((0.0, 0.0, 40.0, 20.0))
    near = Sensed(pos=(5.0, 10.0), heading=0.0, fov_deg=360.0, radius_m=8.0)
    far = Sensed(pos=(32.0, 10.0), heading=0.0, fov_deg=360.0, radius_m=8.0)
    old = cmap.column_of(6.0, 10.0)
    assert old is not None

    _, in_view = cmap.sense(0.0, [near], {"walker": (6.0, 10.0, 1.8)})
    assert in_view == ["walker"] and cmap.levels()[old] > 0
    assert cmap.levels()[cmap.column_of(20.0, 10.0)] == -1  # never observed: unknown

    # the object walks out of range and the observer looks elsewhere: the old column is a ghost
    moved = {"walker": (20.0, 2.0, 1.8)}
    _, in_view = cmap.sense(1.0, [far], moved)
    assert in_view == [] and cmap.levels()[old] > 0 and cmap.ghosts(1.0, moved) == 1

    # the old column is swept again with nothing in it: dimOS carves it back to free ground
    cmap.sense(2.0, [near], moved)
    assert cmap.levels()[old] == 0 and cmap.ghosts(2.0, moved) == 0
    assert cmap.last_touched[old] == 2.0

    # a wedge only touches what it faces
    wedge = ColumnMap((0.0, 0.0, 40.0, 20.0))
    wedge.sense(0.0, [Sensed(pos=(20.0, 10.0), heading=0.0, fov_deg=90.0, radius_m=8.0)], {})
    assert wedge.levels()[wedge.column_of(25.0, 10.0)] == 0
    assert wedge.levels()[wedge.column_of(15.0, 10.0)] == -1


@pytest.fixture(scope="module")
def yard_trace(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    import os

    sys.path.insert(0, str(PITCH))
    from perception_map import episode_trace

    from airtight.contracts import PositionEvent, read_episode_log
    from airtight.sim.runner import run_episode
    from airtight.sim.scenarios import load_fleet, load_sensor_curves, load_site, load_tactic

    site, curves = load_site(), load_sensor_curves()
    # the scenario's jog sits in the charging window; an early phase keeps both drones on patrol
    fleet = load_fleet("2drones")
    tactic = load_tactic("jog").model_copy(update={"phase": 0.1})
    before = os.environ.get("AIRTIGHT_ENGINE")
    os.environ["AIRTIGHT_ENGINE"] = "v0"
    try:
        res = run_episode(site, fleet, tactic, curves, 11, tmp_path_factory.mktemp("trace_logs"))
    finally:
        if before is None:
            del os.environ["AIRTIGHT_ENGINE"]
        else:
            os.environ["AIRTIGHT_ENGINE"] = before
    _, events = read_episode_log(res.log_path)
    tracks: dict[str, list[tuple[float, float, float]]] = {}
    for ev in events:
        if isinstance(ev, PositionEvent):
            tracks.setdefault(ev.object_id, []).append((ev.t, ev.position.x, ev.position.y))
    return site, tactic, episode_trace(site, fleet, tactic, curves, 11), tracks


def test_controller_trace_is_the_logged_episode(yard_trace) -> None:  # type: ignore[no-untyped-def]
    from perception_map import check_trace_against_log

    _, _, trace, tracks = yard_trace
    assert check_trace_against_log(trace, tracks) >= len(trace.frames)
    shifted = {k: [(t, x + 0.01, y) for t, x, y in v] for k, v in tracks.items()}
    with pytest.raises(ValueError, match="not the recorded episode"):
        check_trace_against_log(trace, shifted)


def test_candidate_sets_are_the_top_cells_of_the_agents_own_region(yard_trace) -> None:  # type: ignore[no-untyped-def]
    _, _, trace, _ = yard_trace
    seen = 0
    for frame in trace.frames:
        for aid, a in frame.agents.items():
            if a["mode"] != "patrol" or not a["active"]:
                assert aid not in frame.candidates
                continue
            cells = frame.candidates[aid]
            assert len(cells) >= 1
            priority = (trace.staleness(frame) * trace.weight).ravel()
            assert (priority[cells] > 0).all()
            if not frame.fell_back[aid]:
                assert frame.regions[aid].ravel()[cells].all()
                positive = int(((priority > 0) & frame.regions[aid].ravel()).sum())
                assert len(cells) == max(1, int(trace.top_fraction * positive))
            seen += 1
    assert seen > 0


def test_perception_payload_stays_small(yard_trace) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("dimos.mapping.voxels.impl.packed")
    from perception_map import encode, perceive

    site, tactic, trace, _ = yard_trace
    view = encode(trace, perceive(trace, site.bounds), tactic, site)
    blob = json.dumps(view, separators=(",", ":"))
    # yard_night has 2.5 times the area of the pitch yard, whose whole page is bounded in the build_app test
    assert len(blob) < 200_000
    assert view["n"] == len(trace.frames) == len(view["swept"]) == len(view["pm"]["touch"])
    assert set(view["agents"]) == set(trace.agent_meta) and view["pmax"] > 0
    assert all(len(a["hd"]) == view["n"] == len(a["cand"]) for a in view["agents"].values())


def test_template_has_the_two_views_and_the_zoom_controls() -> None:
    html = (PITCH / "app_template.html").read_text()
    assert 'id="pView" role="tablist"' in html
    assert 'data-view="site"' in html and 'data-view="perceived"' in html
    assert all(f'id="{i}"' in html for i in ("pZoomIn", "pZoomOut", "pZoomReset", "pPrio", "pSeek"))
    assert "addEventListener('wheel'" in html and "prefers-reduced-motion" in html
    assert html.count("__DATA__") == 1
