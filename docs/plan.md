# Airtight: 24-hour build plan (HELD, private repo only)

Three lanes, no shared files. Read sections 1 and 2 together in hour 0. After that each person needs only their own lane and the sync table.

The demo, in order: the red team finds the charging-window attack, the replay runs in dimOS with the Go2 responding, the system proposes a staggered charging schedule, the rescore goes up, and (if lane C's memory lands) the base link is cut live and the fleet still delivers the alarm.

## 0. What is already true at hour 0 (verified 2026-09-19)

- dimOS dev install works at commit `2282a4a52`; the Go2 MuJoCo blueprint runs, the robot walks on `cmd_vel`, the agentic blueprint deploys an MCP server with 15 skills that can be listed and called directly. Lane A step A1 is done except the live LLM turn.
- On WSL2 every blueprint run needs `source scripts/dimos_env.sh` (Mesa adapter override, EGL) and `--rerun-open none`; replays use the bundled `dimos-viewer`, not the Rerun viewer on PATH.
- The MuJoCo child needs ~18 s on an idle box and 5+ minutes on a loaded one. Never run lane B's sweep on the machine lane A is recording on.
- First runs download ~1.5 GB (mujoco_menagerie, a 139 MB model). Budget it on every new machine.
- The shell's `OPENAI_API_KEY` is a placeholder and `OPENAI_BASE_URL` points somewhere unreachable; fix both before C4 and A6. The MCP client has a fixture-backed mock model (`McpClientConfig.model_fixture`) for recording the agent beat without live calls.
- dimOS already ships a scripted person body in the MuJoCo process (`dimos/simulation/mujoco/person_on_track.py`), a `look_out_for` detection skill and `navigate_with_text`. Check each before building A3, A4 and A6.
- All three hour-1 stubs are shipped in this scaffold: the stub sensor curve, the stub `run_episode`, and one hand-written tactic. Lane C adds the other two tactic examples in C0.

## 1. Lanes

One rule: you only edit directories you own. Lanes talk through `contracts/` and nothing else. Nobody imports another lane's internals.

| Path | Owner |
|------|-------|
| `src/airtight/contracts/` | all three agree before any change (schemas, example files, tests) |
| `src/airtight/dimos_lane/` | A: modules, blueprint, calibration, replay |
| `src/airtight/sim/`, `src/airtight/score/` | B: episode runner, sensing, battery, sweep, scorer, report, fix loop |
| `src/airtight/redteam/`, `src/airtight/memory/`, `pitch/` | C: tactics, search, LLM proposer, token accounting, fleet memory, charts, deck, video |
| `data/` | generated, git-ignored except `seeds.json` |

Git: one long-lived branch per lane (`lane-a`, `lane-b`, `lane-c`). Merge to `main` only at the sync points. A merge must pass `uv run pytest tests/contracts`. If you need something from another lane before a sync point, ask for the file; never reach into their code.

dimOS is pinned by commit in `pins.toml`. The swarm autonomy stack in `src/airtight/swarm/` is frozen for the 24 hours: lane B adapts it inside `sim/`, never edits it.

## 2. Contracts (frozen hour 1)

Pydantic models in `src/airtight/contracts/`, one example each in `contracts/examples/`, tests in `tests/contracts/`.

- `site.py`: bounds, perimeter, entry points with ids, asset, response time (defines the critical detection point), docks, fixed sensors, benign routes with class and arrival rate.
- `fleet.py`: agents (id, type drone|go2|guard, speed, endurance, charge time, sensor type), charge policy (threshold, stagger offsets), comms mode, cost per type.
- `tactic.py`: family (charging_window|decoy|blind_spot|comms_cut), entry id, phase in [0,1), speed, waypoints, optional decoy, optional comms event, origin.
- `sensor_curve.py`: per sensor type, range bins, Pd per look per bin, Pfa per look per benign class, field of view, look rate.
- `episode.py`: JSONL header (site hash, fleet hash, tactic, seed) then typed events: position (2 Hz), detection, score, task, battery, comms_graph, alarm_delivered, outcome. `EpisodeResult` is what `run_episode` returns.
- `report.py`: per configuration ROC points, Pd at the operating point with interval, worst tactic and its Pd, cost per hour, coverage gap seconds per hour, human decisions per hour, paired deltas against baseline; plus the conditions block every slide must quote.

Owned by B, called by C: `airtight.sim.runner.run_episode(site, fleet, tactic, sensor_curves, seed, log_dir) -> EpisodeResult`.
Owned by C, called by B and wrapped by A: `airtight.memory.FleetMemory` protocol (`observe`, `delta(since_version, byte_budget)`, `merge`, `query`, `version`).

## 3. Lane A: dimOS

| Step | Hours | Do | Done when |
|---|---|---|---|
| A0 | 0–1 | Contracts with the team. `uv sync --extra dev`, `./scripts/check_pins.sh`, `uv run dimos run airtight.airtight-site` lists `site_status`. Set a real OpenAI key and fix `OPENAI_BASE_URL`. | Hello blueprint runs from this repo. |
| A1 | 1–2 | `dimos --simulation --rerun-open none run unitree-go2-agentic`; `dimos mcp list-tools`; `dimos mcp call get_battery_soc`; `dimos agent-send "walk forward"`. | The Go2 moves from a skill call and the LLM turn completes. |
| A2 | 2–4 | Try MuJoCo and DimSim with the empty scene. Pick one on three tests: near real time on your box, a body the detector sees, a body's pose scriptable over time (start from `person_on_track.py`). | Gate H4: simulator chosen, else fallback F1. |
| A3 | 4–6 | Generate the yard from `site.json`: perimeter walls, asset marker, dock marker, an intruder body (G1 on MuJoCo), two benign objects. | Go2 camera frame shows the intruder inside the yard. |
| A4 | 6–9 | Calibration: intruder at 2, 4, 6, 8, 12, 16, 20, 30 m and three bearings, 30+ frames each; run the dimOS detector with a text query; hit or miss per frame; repeat for benign objects; fit logistic Pd by range; cache detector calls. | `data/sensor_curve.json` v1 to B at hour 9. |
| A5 | 9–12 | Modules as thin wrappers: AllocatorModule (CBBA), GateModule (propose, decide, pending skills), SimFleetModule (kinematic drones, topics prefixed by robot id), DimosBackend (goto calls Go2 navigation, pose reads odometry). | Each module starts alone. |
| A6 | 12–14 | Blueprint `airtight-site`: Go2 connection, MCP server and client, the four modules. Skills `dispatch_verify(x, y)`, `fleet_status()`. | "check the north gate" produces a task, an auction and a moving Go2. |
| A7 | 14–16 | If C's memory passed its gate: FleetMemoryModule and a `recall(query)` skill. Else start A8 early. | `recall` returns entries with age. |
| A8 | 16–19 | Replay runner: read an episode log, script the intruder along the logged path, drones as Rerun markers, dispatch the Go2 at the logged alarm time. Top failure and the same seed after the fix. | Two clips: the miss, the catch. |
| A9 | 19–20 | Clean takes to C. | Clips in `pitch/`. |

Calibration assumptions to state: only the Go2 camera is measured; the drone camera reuses the curve rescaled by apparent target size; the radar curve is parametric.

## 4. Lane B: fast sim and score

Build on `src/airtight/swarm/`: `tests/swarm/road_harness.py` and `swarm/autonomy/training.py` drive the kinematic backend, `swarm/autonomy/allocator.py` (CBBA), `swarm/autonomy/decomposer.py` (HTN), Voronoi coverage, `swarm/edge/sensing.py` (multi-target measurement synthesizer), `swarm/hol/gate.py` (silence auto-denies). The allocator types agents as `DroneState`; adapting guards and the Go2 is B's first design call.

| Step | Hours | Do | Done when |
|---|---|---|---|
| B0 | 0–1 | Contracts. Stub `run_episode` is already shipped; read it. | Stub understood. |
| B1 | 1–3 | Run the road harness. Site loader, Intruder actor (follows tactic waypoints, reaching the asset is a miss), Benign actors (Poisson arrivals, a class each). | Episode runs with actors moving and writes a log. |
| B2 | 3–5 | Sensing: each sensor looks at its rate; in range and in FOV, Bernoulli draw from the Pd curve; benign objects fire at their class rate. Track score per object: add ln(pd/pfa) on a hit, ln((1-pd)/(1-pfa)) on an in-view miss, clamped. Truth association (state as a limit). | Intruder score rises walking past a drone; empty patch stays flat. |
| B3 | 5–7 | Battery and docks: endurance, charge time, dock count, threshold, stagger. Coverage recomputes over active agents. Steady state cached by config hash; an episode loads the snapshot at the tactic's phase and runs only the intrusion window. | Synchronized-charging config shows a coverage dip at a known phase. |
| B4 | 7–8 | Events and the criterion. A fixed low "investigate" score creates a verify task assigned by CBBA (nearest capable agent if CBBA wiring exceeds an hour; say so). Full score series logged; no alarm threshold in the sim. Timely = alarm before t_reach_asset − response_time. | Gate H8: real `run_episode` replaces the stub. |
| B5 | 8–10 | Offline scorer: per threshold, timely-detection fraction and false alarms per benign hour; interpolate Pd at 1 FA/h; Wilson per point, bootstrap for the interpolated value. Swap in A's curve at hour 10. | ROC from 50 episodes. |
| B6 | 10–12 | Sweep with a process pool: ~12 configs (1–4 drones, Go2 on/off, guard on/off, sync/staggered), C's top tactics, the committed seed list. | Gate H12: first `report.json`; three worst failure logs to A. |
| B7 | 12–14 | Cost per hour per type (assumption table for C), coverage gap s/h, decisions per hour at the operating point, paired deltas with intervals. | Cost-versus-detection points in the report. |
| B8 | 14–16 | Fix loop: search stagger offsets maximizing worst-tactic Pd at equal cost; rescore on the same seeds; let C re-attack the fixed config once. | Before and after in the report. Sim features frozen at H16. |
| B9 | 16–18 | Only if memory passed: three comms modes on the two headline configs, alarm routed through the channel. | Three-row comparison. |
| B10 | 18–20 | Final sweep at 200 seeds per cell. Freeze `report.json`. Final replay logs to A by hour 18. | Report frozen. |

## 5. Lane C: red team, memory, pitch

See `docs/lanes/lane-c.md` for the step-by-step version.

| Step | Hours | Do | Done when |
|---|---|---|---|
| C0 | 0–1 | Contracts. Two more hand-written tactic examples (decoy, blind_spot). | Examples merged. |
| C1 | 1–3 | Parameterize three families. Validator: starts outside the perimeter, stays in bounds, speed under the cap. | Random valid tactics on demand. |
| C2 | 3–6 | Search against the stub: miss rate over 20 shared seeds; 200 random per family, keep 10, refine by perturbation. | Search runs end to end on the stub. |
| C4a | 6–8 | LLM proposer: site digest, patrol summary, per-family results table in; primitive-built tactic instances out; schema-validate; cache by prompt hash; log tokens and cost. | Proposer returns valid tactics on the stub. |
| C3 | 8–10 | Rerun on the real sim. Baseline Pd should land in 0.6–0.9; if not, B adjusts range or site size now. Scenario fixed after this. | Charging-window tactic finds the sync dip. |
| C4b | 10–11 | Proposer against the real sim; survivors into the search. | One LLM-proposed tactic survives into the top set. |
| C5 | 11–14 | Fleet memory: coverage cells merge by max, claims by newest, evidence as a deduplicated set with summed score. Property tests for order independence and idempotence. Deltas respect the byte budget, newest first. | Gate H14: tests pass, or the link-cut beat is dropped. |
| C6 | 14–15 | Token accounting: naive planner on 20 episodes vs propose-then-search per configuration. | One chart. |
| C7 | 15–16 | Comms tactic: cut the base link at a chosen time. | Valid against B9. |
| C8 | 16–20 | Charts from `report.json` by script; eight-slide deck; demo script; track write-ups; video with A's clips. | Draft video at hour 20. |

## 6. Sync points (ten minutes, standing; merge to main only here)

| Hour | Hand-offs | Decision |
|---|---|---|
| 1 | Contracts and stubs merged. | Keys set, every machine runs the hello blueprint. |
| 4 | A reports the simulator choice. | Fallback F1 or not. |
| 5 | C reports stub-search difficulty. | Early warning for the hour-10 difficulty gate. |
| 8 | B's real `run_episode` replaces the stub. | C moves search to the real sim. |
| 10 | A's sensor curve into B. C and B agree on difficulty. | Scenario parameters fixed. |
| 12 | First `report.json`. Failure logs to A. | Charging-window story visible? Else F2. |
| 14 | C's memory gate. | Link-cut beat in or out. |
| 16 | Sim features frozen. | No tuning after this. |
| 18 | Final logs to A. | |
| 20 | Everything frozen. | Record 20 to 23, submit. |

## 7. Fallbacks

- F1: dimOS sim won't run by hour 4. A builds modules and blueprint against dimOS replay blueprints, calibrates on recorded Go2 footage plus stills, replays come from a Rerun script over the episode log. Pitch leans on modules and skills.
- F2: the charging-window attack doesn't separate configs at hour 12. Lead with whichever tactic does (decoy next) and make the fix the matching one (a held-back agent).
- F3: memory misses hour 14. Drop the link-cut beat; end on the rescore; memory goes on the roadmap slide.
- F4: detector too slow or costly. Fewer range bins, 15 frames each, or a local detector. State the sample size.
- F5: sweep too slow at hour 18. 100 seeds and six configs; report the wider intervals as they are.

## 8. Two-person mode

Person 1 takes lane A unchanged. Person 2 takes lane B plus C1–C3 (search only, no LLM until hour 12). Person 1 adds the proposer and token accounting after A6 at hour 14. Memory and the link-cut beat are dropped. Deck and video shared from hour 18; sim features freeze at hour 15.

## 9. Rules that protect the result

- One seed list, `data/seeds.json`, committed, used by every configuration and every before-and-after.
- Scenario parameters fixed at hour 10, sim features at hour 16. No tuning after either.
- Every number on a slide is produced by a script reading `report.json`. No hand-typed numbers.
- The score line always carries its conditions: operating point, interval, worst tactic, adversary knowledge, calibration method, and that the sweep uses a reduced-order detection model.
- When the fix is shown, show the adversary's re-attack number next to it.
