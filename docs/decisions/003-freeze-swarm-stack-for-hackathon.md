# Decision 003: Freeze Swarm Stack For Hackathon

## Context
- `src/airtight/swarm/` holds the multi-agent autonomy stack: CBBA allocation, HTN decomposition, Voronoi coverage, behaviour-tree execution, the multi-sensor measurement synthesizer, UKF and JPDA tracking with Dempster-Shafer classification, and the human approval gate, with its own tests under `tests/swarm/`.
- Three lanes work in parallel for 24 hours; the stack is a dependency of lane B's simulator and of lane A's modules.
- The allocator types agents as `DroneState`, and the mixed fleet needs guards and a ground robot in the same auction.

## Decision
The stack is frozen for the hackathon: no lane edits `src/airtight/swarm/`; adaptation for the mixed fleet, the site twin and the battery model happens in `src/airtight/sim/` and `src/airtight/dimos_lane/` by wrapping or subclassing.

## Reason
- A shared, edited dependency breaks the no-shared-files rule and makes the hour-8 and hour-12 hand-offs unreproducible.
- Wrapping keeps the interview-critical core unchanged and reviewable; the adapter is small and lane-local.
- Its 96 tests stay green as a regression guard for every merge.

## Consequences
- `tests/swarm/` runs in every merge gate alongside `tests/contracts/`.
- Lane B's first design call is the agent adapter: a guard or Go2 expressed as a `DroneState` with its own kinematics and sensor, or a generalized agent protocol written in `sim/`.
- Naming-convention lint rules are relaxed for the stack's matrix-heavy code in `pyproject.toml`.
