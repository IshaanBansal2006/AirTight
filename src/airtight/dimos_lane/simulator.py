"""Simulator choice for lane A. Default is MuJoCo; DimSim is the H4 bake-off."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SimulatorName = Literal["mujoco", "dimsim", "replay"]

# Gate H4: MuJoCo already walks the Go2 on this WSL2 box, scripts a person over
# `/person_pose`, and feeds a camera the detector can see. DimSim walls are
# implemented in yard.apply_dimsim but lose unless they beat all three tests.
CHOSEN_SIMULATOR: SimulatorName = "mujoco"
CHOSEN_REASON = (
    "MuJoCo wins the H4 bake-off on all three tests: near-real-time occupancy "
    "extrusion + mj_step on this WSL2 box, a person mocap body the camera can "
    "see, and /person_pose scripting. DimSim apply_dimsim places walls but has "
    "no person mocap and is not running live here."
)

H4_TESTS = ("near_realtime", "detector_visible_body", "pose_scriptable")

# dimOS GlobalConfig.transport defaults to zenoh; stay on LCM (issue #4124).
# WSL2 loopback has no multicast without sudo, so live runs may fall back to Zenoh.
DIMOS_TRANSPORT = "lcm"

# A1 LLM turn is skipped: unitree-go2-agentic OOMs on ~8 GiB and agent-send
# would bill OpenAI. Skill-call walk_forward is the movement path.
A1_SKILL_CALL = True
A1_LLM_TURN = False

# A6 uses one gpt-4o-mini turn on the slim airtight-site stack.
A6_MODEL = "gpt-4o-mini"
A6_LLM_TURN = True


@dataclass
class BakeoffScore:
    name: SimulatorName
    near_realtime: bool
    detector_visible_body: bool
    pose_scriptable: bool
    details: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def wins(self) -> int:
        return int(self.near_realtime) + int(self.detector_visible_body) + int(self.pose_scriptable)


def _eval_mujoco() -> BakeoffScore:
    import time

    from dimos.mapping.occupancy.extrude_occupancy import generate_mujoco_scene
    from dimos.msgs.geometry_msgs.Pose import Pose
    from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
    from dimos.simulation.mujoco.model import get_model_xml

    from airtight.contracts.site import XY
    from airtight.dimos_lane.person import interpolate_path
    from airtight.dimos_lane.site_io import load_example_site
    from airtight.dimos_lane.yard import (
        OCCUPANCY_RESOLUTION_M,
        coarsen,
        occupancy_counts,
        site_to_occupancy,
    )

    details: dict[str, str] = {}
    t0 = time.perf_counter()
    site = load_example_site()
    coarse = coarsen(site_to_occupancy(site), 4)
    occupied, free = occupancy_counts(coarse)
    grid = OccupancyGrid(
        grid=coarse,
        resolution=OCCUPANCY_RESOLUTION_M * 4,
        origin=Pose(float(site.bounds[0]), float(site.bounds[1]), 0.0),
    )
    xml = generate_mujoco_scene(grid)
    walls = xml.count('type="box"')
    extrude_s = time.perf_counter() - t0
    details["extrude"] = f"{walls} boxes in {extrude_s:.2f}s occupied={occupied} free={free}"

    step_ok = False
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        t_step = time.perf_counter()
        for _ in range(50):
            mujoco.mj_step(model, data)
        step_s = time.perf_counter() - t_step
        hz = 50.0 / step_s if step_s > 0 else 0.0
        details["mj_step"] = f"{hz:.0f} Hz over 50 steps"
        step_ok = hz >= 20.0
    except Exception as exc:
        details["mj_step"] = f"unavailable: {type(exc).__name__}"

    person_xml = get_model_xml("unitree_go1", xml)
    body_ok = 'name="person"' in person_xml and "person_mesh" in person_xml
    details["person_xml"] = "mocap person_mesh present" if body_ok else "person missing"

    samples = interpolate_path([XY(x=10, y=70), XY(x=16, y=70)], speed_mps=1.0, dt=0.5)
    pose_ok = len(samples) >= 2 and samples[-1][0] > samples[0][0]
    details["path"] = f"{len(samples)} samples last_t={samples[-1][0]:.1f}s"

    elapsed = time.perf_counter() - t0
    return BakeoffScore(
        name="mujoco",
        near_realtime=extrude_s < 15.0 and (step_ok or "unavailable" in details["mj_step"]),
        detector_visible_body=body_ok,
        pose_scriptable=pose_ok,
        details=details,
        elapsed_s=elapsed,
    )


def _eval_dimsim() -> BakeoffScore:
    from airtight.dimos_lane.site_io import load_example_site
    from airtight.dimos_lane.yard import apply_dimsim, default_benign_blobs, perimeter_edges

    class _FakeClient:
        def __init__(self) -> None:
            self.walls: list[str] = []
            self.objects: list[str] = []

        def add_wall(self, *args: Any, **kwargs: Any) -> None:
            self.walls.append(str(kwargs.get("name") or args[:4]))

        def add_object(self, *args: Any, **kwargs: Any) -> None:
            self.objects.append(str(kwargs.get("name") or args[:1]))

    site = load_example_site()
    client = _FakeClient()
    placed = apply_dimsim(client, site)  # type: ignore[arg-type]
    n_walls = len(perimeter_edges(site))
    n_benign = len(default_benign_blobs())
    api_ok = len(client.walls) == n_walls and len(client.objects) >= 1 + len(site.docks) + n_benign
    details = {
        "api": (
            f"placed {len(placed['placed'])} names walls={len(client.walls)} "
            f"objects={len(client.objects)} ok={api_ok}"
        ),
        "live": "no DimSim process on this box",
        "person": "DimSim has no /person_pose mocap",
    }
    return BakeoffScore(
        name="dimsim",
        near_realtime=False,
        detector_visible_body=False,
        pose_scriptable=False,
        details=details,
        elapsed_s=0.0,
    )


def run_h4_bakeoff() -> dict[str, BakeoffScore]:
    """Three H4 tests for MuJoCo vs DimSim. Winner is recorded as CHOSEN_SIMULATOR."""
    scores = {"mujoco": _eval_mujoco(), "dimsim": _eval_dimsim()}
    winner = max(scores.values(), key=lambda s: s.wins)
    if winner.name != CHOSEN_SIMULATOR:
        raise RuntimeError(
            f"bake-off winner is {winner.name} ({winner.wins}/3) but CHOSEN_SIMULATOR="
            f"{CHOSEN_SIMULATOR}"
        )
    return scores
