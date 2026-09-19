from __future__ import annotations

import inspect
from types import SimpleNamespace

from airtight.dimos_lane.blueprint import A6_MODEL, SITE_AGENT_PROMPT, build_airtight_site
from airtight.dimos_lane.calibration import BEARINGS_DEG, FRAMES_PER_CELL, RANGE_BINS_M, read_looks
from airtight.dimos_lane.calibration.capture import (
    VEHICLE_RANGES_M,
    planned_poses,
    run_live_sweep,
)
from airtight.dimos_lane.calibration.fit import curves_from_looks
from airtight.dimos_lane.openai_env import ensure_openai_key
from airtight.dimos_lane.simulator import A6_LLM_TURN, A6_MODEL as SIM_A6_MODEL


def test_run_live_sweep_covers_bins_and_bearings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = tmp_path / "looks.jsonl"
    placed: list[tuple[float, float]] = []
    grabs = {"n": 0}

    def grab() -> SimpleNamespace:
        grabs["n"] += 1
        return SimpleNamespace(data=f"frame-{grabs['n']}".encode())

    def place(x: float, y: float) -> None:
        placed.append((x, y))

    def detect(image: object, query: str = "person", detector_name: str = "owlv2") -> tuple[bool, int, str]:
        return query == "person", int(query == "person"), "fake-owl"

    looks = run_live_sweep(
        cache,
        grab_frame=grab,
        place_xy=place,
        detect=detect,
        frames_per_cell=FRAMES_PER_CELL,
        settle_s=0.0,
    )
    person_cells = len(RANGE_BINS_M) * len(BEARINGS_DEG)
    vehicle_cells = len(VEHICLE_RANGES_M)
    assert len(looks) == (person_cells + vehicle_cells) * FRAMES_PER_CELL
    assert len(read_looks(cache)) == len(looks)
    person = [lk for lk in looks if lk.cls == "person"]
    vehicle = [lk for lk in looks if lk.cls == "vehicle"]
    assert {lk.range_m for lk in person} == set(RANGE_BINS_M)
    assert {lk.bearing_deg for lk in person} == set(BEARINGS_DEG)
    assert all(lk.hit for lk in person)
    assert not any(lk.hit for lk in vehicle)
    assert all(lk.detector == "fake-owl" for lk in looks)
    assert len(placed) == person_cells + vehicle_cells


def test_run_live_sweep_pd_falls_with_range(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = tmp_path / "looks.jsonl"
    last_range = {"m": 2.0}

    def place(x: float, y: float) -> None:
        # planned person poses: range is recovered from x relative to dock
        last_range["m"] = ((x - 10.0) ** 2 + (y - 70.0) ** 2) ** 0.5

    def detect(image: object, query: str = "person", detector_name: str = "owlv2") -> tuple[bool, int, str]:
        if query != "person":
            return False, 0, "fake-owl"
        hit = last_range["m"] <= 8.0
        return hit, int(hit), "fake-owl"

    looks = run_live_sweep(
        cache,
        grab_frame=lambda: SimpleNamespace(data=b"img"),
        place_xy=place,
        detect=detect,
        frames_per_cell=30,
        settle_s=0.0,
        classes=("person",),
    )
    curves = curves_from_looks(looks)
    go2 = curves.curves["go2_camera"]
    assert go2.pd_per_look[0] > go2.pd_per_look[-1]
    assert go2.pd_per_look[0] == 1.0
    assert go2.pd_per_look[-1] == 0.0


def test_planned_poses_match_range_bearing() -> None:
    poses = planned_poses((10.0, 70.0))
    assert len(poses) == 8 * 3
    front = next(p for p in poses if p[0] == 2.0 and p[1] == 0.0)
    assert abs(front[2] - 12.0) < 1e-6
    assert abs(front[3] - 70.0) < 1e-6


def test_airtight_site_is_slim_not_agentic() -> None:
    src = inspect.getsource(build_airtight_site)
    assert "unitree_go2_agentic" not in src
    assert "McpClient" in src
    assert "WalkModule" in src
    assert A6_MODEL == "gpt-4o-mini"
    assert SIM_A6_MODEL == "gpt-4o-mini"
    assert A6_LLM_TURN is True
    assert "check_north_gate" in SITE_AGENT_PROMPT


def test_ensure_openai_key_from_yaml(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPEN_AI_KEY: sk-test-not-a-real-key\n")
    assert ensure_openai_key(env_path=env) is True
    assert os.environ["OPENAI_API_KEY"] == "sk-test-not-a-real-key"
