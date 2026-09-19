# Decision 007: Fleet Memory CRDT

## Context
- The link-cut demo beat needs replicas that keep merging what they know when the base link is gone and reconverge when it returns.
- Three entry kinds were named: coverage cells, task claims, detection evidence.
- Bandwidth on a degraded link is the constraint, so a delta must fit a byte budget.

## Decision
`memory/store.py` is a state-based CRDT. Coverage cells are max-registers on last-seen time, claims are last-writer-wins on time with the agent id as a deterministic tie-break, evidence is a grow-only set keyed by evidence id whose conflicts resolve by a total order on the serialized item; an object's score is the sum over its evidence. `delta(since_version, byte_budget)` ships entries changed after a version, newest first, as JSONL cut to the budget; `merge` re-stamps changed entries so they keep propagating.

## Reason
- Each rule is commutative, associative and idempotent by construction, so order and duplication cannot matter; hypothesis property tests assert this over random item sets and shuffles.
- Newest-first under a budget delivers the freshest claims and sightings when the link is poor, which is what the responder needs.
- A deterministic tie-break on evidence was forced by the property tests: a first-wins rule failed order independence when two entries shared an id with different content.

## Consequences
- 96 property examples per test at hypothesis defaults; `tests/memory/` runs in the merge gate.
- Lane B routes alarms through `delta`/`merge` under the three comms modes; lane A wraps the store in a dimOS module with a `recall` skill.
- Version numbers are per replica; a receiver tracks the last version it saw from each peer.
