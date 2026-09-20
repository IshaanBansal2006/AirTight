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
git clone -b drone-autonomy https://github.com/IshaanBansal2006/dimos-airtight.git dimos   # private working copy of dimOS
git clone <this repo> AirTight && cd AirTight
uv sync --extra dev
./scripts/check_pins.sh
./scripts/dimos_patch.sh apply
uv run pytest
```

dimOS needs its system packages first; run `scripts/install.sh` from the dimos checkout
once, or follow its `docs/installation/`. On WSL2 source `scripts/dimos_env.sh` before any
blueprint run. The dimos checkout is our private working copy on branch `drone-autonomy`;
its `AIRTIGHT.md` says what may change there and what may not. Changes to dimOS go to that
repo as pull requests, not into this one.

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

## Commands

Everything below runs from the repo root after `uv sync --extra dev`. The demo scenario lives in
`scenarios/logistics_yard/`; pass `--engine v0` to use the simulation engine instead of the stub.

```bash
uv run airtight-redteam difficulty --engine v0 --site scenarios/logistics_yard/site.json --fleet scenarios/logistics_yard/fleets/d2_go2_guard_sync.json --curves scenarios/logistics_yard/sensor_curve.json --config scenarios/logistics_yard/redteam_config.json --workers 8
uv run airtight-redteam search     --engine v0 <same scene flags> --out data/tactics          # worst tactic per family, replay logs for each
uv run airtight-redteam propose    <scene flags> --prior data/tactics --mock                    # LLM proposals; drop --mock for one real call under the cap
uv run airtight-sweep --engine v0 --tactics-dir data/tactics --n-seeds 200 --quiet-seeds 20     # twelve fleets x worst tactics -> data/report.json
scripts/rebuild_pitch.sh data/report.json data/tactics                                          # charts, clips, deck, write-up, rendered deck
uv run python pitch/render_replay.py <episode.jsonl>                                            # one episode as MP4
```

## Results archive

`data/` is ignored by git and holds raw episode logs. Everything a run measures is copied into
`results/<run>/` with `scripts/archive_run.sh <run> data/<run>` and committed: search elites and
every scored tactic (`tactics/top_<family>.json`), the sweep report and its per-configuration
detail (`report.json`, `report_detail/`), difficulty and campaign reports, min-max iterations,
replay logs for each family's best tactic, the LLM ledger, and a `manifest.json` with the git
commit and scenario hashes. Prune later by deleting what is not needed; the manifest lists every
file.
