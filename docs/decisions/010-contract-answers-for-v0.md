# Decision 010: Contract Answers For The v0 Engine

## Context
- Lane B's v0 engine (PR #5) raised three contract questions: tactic speed has no upper bound in the schema; with no task time at the asset the critical detection point lands mid-path; with a task time and one asset the patrol camps on the asset.
- Lane C's first run of the difficulty gate through v0 on the demo site scored mean detection 0.02, far below the 0.6 to 0.9 band.

## Decision
The speed cap stays in the red-team validator (2.5 m/s, decision 004), not in the contract. Task time at the asset stays zero: timeliness is alarm before `t_reach − response_time`. The scenario is brought into band by tuning the site's response time and sensor ranges, not by adding assets. The demo site's response time is set from the v0 difficulty sweep recorded below.

## Reason
- A schema bound would make every tactic file carry a scenario constant; the validator already rejects fast intruders with a reason, and the LLM proposer is gated by the same validator.
- Zero task time keeps the criterion the plan and the decision docs already state, and keeps the CDP comparable across entries; a task time would need multiple assets to avoid camping, which is out of scope for 24 hours.
- Response time is the one scenario knob that moves the CDP for every entry at once and has a real-world meaning a buyer understands.

## Consequences
- `scenarios/logistics_yard/site.json` response time is the value chosen in the sweep (see the amendment in decision 008); the conditions line quotes it.
- Lane C searches call the runner with `full_log=False` and write full logs only for each family's best tactic on the first three seeds (`data/tactics/replays/`), which is what lane A's replay and the failure clips consume.
- Part 2 of the engine (battery, phase, comms, decoy reaction) is expected to lower detection again at the charging window; the gate is rerun when it lands.
