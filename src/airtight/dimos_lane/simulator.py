"""Simulator choice for lane A. Default is MuJoCo; DimSim is the H4 bake-off."""

from __future__ import annotations

from typing import Literal

SimulatorName = Literal["mujoco", "dimsim", "replay"]

# Gate H4: MuJoCo already walks the Go2 on this WSL2 box, scripts a person over
# `/person_pose`, and feeds a camera the detector can see. DimSim walls are
# implemented in yard.apply_dimsim but lose unless they beat all three tests.
CHOSEN_SIMULATOR: SimulatorName = "mujoco"
CHOSEN_REASON = (
    "MuJoCo is the H4 pick: Go2 agentic sim runs on this WSL2 box, the person "
    "mocap body is pose-scriptable over /person_pose, and occupancy extrusion "
    "builds the yard from site.json. DimSim remains a fallback path."
)

# dimOS GlobalConfig.transport defaults to zenoh; stay on LCM (issue #4124).
DIMOS_TRANSPORT = "lcm"
