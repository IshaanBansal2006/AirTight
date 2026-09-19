# airtight

Simulation tooling for evaluating the security posture of a commercial site when it is
patrolled by a mixed fleet of drones, a ground robot and human guards. Built on
[dimOS](https://github.com/dimensionalOS/dimos) for the robot runtime and on
[drone-swarm-autonomy](https://github.com/IshaanBansal2006/drone-swarm-autonomy) for the
multi-agent autonomy stack.

## Pre-existing code

Two dependencies pre-date this repository and are pulled in unmodified as editable path
dependencies of sibling checkouts (commits recorded in `pins.toml`):

- `dimos`: Dimensional's robot operating system (Apache-2.0).
- `drone-swarm-autonomy`: task allocation, sensing synthesis, tracking and the human
  approval gate (MIT, same author).

Everything under `src/airtight/` is new.

## Setup

```bash
cd ~/projects
git clone https://github.com/dimensionalOS/dimos.git && git -C dimos checkout 2282a4a52
git clone https://github.com/IshaanBansal2006/drone-swarm-autonomy.git && git -C drone-swarm-autonomy checkout 0b9bd57
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
| `src/airtight/dimos_lane/` | lane A | dimOS modules, the `airtight-site` blueprint, replay |
| `src/airtight/sim/` | lane B | headless episode runner |
| `src/airtight/score/` | lane B | offline scoring and reports |
| `src/airtight/redteam/` | lane C | adversary tactics and search |
| `src/airtight/memory/` | lane C | fleet memory |
| `pitch/` | lane C | charts, deck, video |
| `data/` | generated | only `seeds.json` is tracked |

Lanes communicate through `contracts/` only. Each lane works on its own branch and merges
to `main` when `uv run pytest tests/contracts` passes.

Run the hello blueprint to confirm the entry point is discovered:

```bash
uv run dimos run airtight.airtight-site
```
