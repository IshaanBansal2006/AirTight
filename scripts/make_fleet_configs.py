"""Generate the sweep's fleet configurations. Edit the tables here, rerun, commit the JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "scenarios" / "logistics_yard" / "fleets"

DRONE = {
    "speed_mps": 8.0,
    "endurance_s": 1500.0,
    "charge_time_s": 2400.0,
    "sensor_type": "drone_camera",
}
GO2 = {
    "speed_mps": 1.2,
    "endurance_s": 5400.0,
    "charge_time_s": 3600.0,
    "sensor_type": "go2_camera",
}
GUARD = {"speed_mps": 1.4, "endurance_s": 28800.0, "charge_time_s": 0.0, "sensor_type": "human_eye"}
COST_PER_HOUR = {"drone": 7.0, "go2": 9.0, "guard": 32.0}
THRESHOLD = 0.2

SWEEP: list[tuple[int, bool, bool, bool]] = [
    (1, True, True, False),
    (1, True, True, True),
    (2, True, True, False),
    (2, True, True, True),
    (3, True, True, False),
    (3, True, True, True),
    (4, True, True, False),
    (4, True, True, True),
    (2, False, True, True),
    (2, True, False, True),
    (2, False, False, True),
    (0, True, True, False),
]


def name(n_drones: int, go2: bool, guard: bool, stagger: bool) -> str:
    parts = [
        f"d{n_drones}",
        "go2" if go2 else "nogo2",
        "guard" if guard else "noguard",
        "stagger" if stagger else "sync",
    ]
    return "_".join(parts)


def build(n_drones: int, go2: bool, guard: bool, stagger: bool) -> dict:
    agents = [{"id": f"drone_{i + 1}", "type": "drone", **DRONE} for i in range(n_drones)]
    if go2:
        agents.append({"id": "go2_1", "type": "go2", **GO2})
    if guard:
        agents.append({"id": "guard_1", "type": "guard", **GUARD})
    offsets: dict[str, float] = {}
    if stagger and n_drones > 1:
        cycle = DRONE["endurance_s"] + DRONE["charge_time_s"]
        offsets = {f"drone_{i + 1}": round(i * cycle / n_drones, 1) for i in range(n_drones)}
    return {
        "name": name(n_drones, go2, guard, stagger),
        "agents": agents,
        "charge_policy": {"threshold_frac": THRESHOLD, "stagger_offsets_s": offsets},
        "comms_mode": "central_perfect",
        "cost_per_hour_by_type": {
            t: c for t, c in COST_PER_HOUR.items() if any(a["type"] == t for a in agents)
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in SWEEP:
        cfg = build(*spec)
        if not cfg["agents"]:
            print(f"skipping {cfg['name']}: no agents", file=sys.stderr)
            continue
        (OUT / f"{cfg['name']}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (OUT / "sweep.json").write_text(
        json.dumps(
            {"baseline": name(2, True, True, False), "configs": [name(*s) for s in SWEEP]}, indent=2
        )
        + "\n"
    )
    print(f"{len(SWEEP)} fleet configs written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
