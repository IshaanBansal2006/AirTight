# airtight

Simulation tooling for evaluating the security posture of a commercial site when it is
patrolled by a mixed fleet of drones, a ground robot and human guards. Built on
[dimOS](https://github.com/dimensionalOS/dimos) for the robot runtime.

`src/airtight/swarm/` is the multi-agent autonomy stack: task allocation (CBBA), HTN
decomposition, Voronoi coverage, behaviour-tree execution, a multi-sensor measurement
synthesizer, UKF tracking with Dempster-Shafer classification, and a human approval gate.

## Setup

```bash
cd ~/projects
git clone https://github.com/dimensionalOS/dimos.git && git -C dimos checkout 2282a4a52
git clone <this repo> AirTight && cd AirTight
uv sync --extra dev
./scripts/check_pins.sh
uv run pytest
```

dimOS needs its system packages first; run `scripts/install.sh` from the dimos checkout
once, or follow its `docs/installation/`. On WSL2 source `scripts/dimos_env.sh` before any
blueprint run.

## Layout

| Path | Owner | Contents |
|------|-------|----------|
| `src/airtight/contracts/` | all | Pydantic schemas and one example file per schema |
| `src/airtight/swarm/` | all (frozen) | allocation, coverage, sensing, tracking, approval gate |
| `src/airtight/dimos_lane/` | lane A | dimOS modules, the `airtight-site` blueprint, replay |
| `src/airtight/sim/` | lane B | headless episode runner |
| `src/airtight/score/` | lane B | offline scoring and reports |
| `src/airtight/redteam/` | lane C | adversary tactics and search |
| `src/airtight/memory/` | lane C | fleet memory |
| `pitch/` | lane C | charts, deck, video |
| `data/` | generated | only `seeds.json` is tracked |

Lanes communicate through `contracts/` only. Each lane works on its own branch and merges
to `main` when `uv run pytest tests/contracts` passes.

Run the site blueprint to confirm the entry point is discovered:

```bash
uv run dimos run airtight.airtight-site
```
