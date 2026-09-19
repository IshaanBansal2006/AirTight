from __future__ import annotations

from airtight.dimos_lane.simulator import (
    CHOSEN_SIMULATOR,
    H4_TESTS,
    run_h4_bakeoff,
)


def test_h4_bakeoff_picks_mujoco() -> None:
    scores = run_h4_bakeoff()
    assert set(scores) == {"mujoco", "dimsim"}
    mujoco = scores["mujoco"]
    dimsim = scores["dimsim"]
    assert mujoco.wins == 3
    assert mujoco.near_realtime
    assert mujoco.detector_visible_body
    assert mujoco.pose_scriptable
    assert dimsim.wins < mujoco.wins
    assert CHOSEN_SIMULATOR == "mujoco"
    assert H4_TESTS == ("near_realtime", "detector_visible_body", "pose_scriptable")
    assert "boxes" in mujoco.details["extrude"]
    assert "person" in mujoco.details["person_xml"]
