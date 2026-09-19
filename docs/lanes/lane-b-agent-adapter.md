# Lane B's first design call: guards and the Go2 in one auction (HELD)

The allocator in `swarm/autonomy/allocator.py` types every bidder as `DroneState` and its capability
check reads drone fields. The mixed fleet is the thesis, so the adapter is the first thing lane B
builds. Options, with the lead's default at the end so B can start without a meeting.

## Option 1: everything is a `DroneState`

Guards and the Go2 are drones with odd parameters: altitude pinned to zero, speed 1.2 to 1.4 m/s,
guard endurance eight hours and charge time zero, capability strings that only ground tasks match.

- Pros: zero changes to the allocator, CBBA runs unchanged, done in an hour, the swarm tests stay green.
- Cons: every downstream consumer must remember that "drone" may be a person; the battery model
  applies to a guard unless special-cased; the path score in `TimeDiscountedScore` assumes straight
  flight, so ground agents bid as if they could cross the yard in a line.

## Option 2: an `Agent` protocol in `sim/` with per-type kinematics, and a thin shim to `DroneState`

`sim/agents.py` defines the agent (id, type, position, speed, energy, sensor) and a `to_drone_state()`
shim used only at the auction boundary; movement, battery and sensing run on the typed agent.

- Pros: ground agents move along the yard, not through it; the battery model is per type; the report's
  per-type costs and decisions map cleanly; it is the design an interviewer expects.
- Cons: two representations to keep in sync; the shim still lies to the score function about
  travel time unless B overrides `PathScore` (it is a protocol with one method, so that is cheap).

## Option 3: generalize the allocator itself

Change `DroneState` to an agent base type inside `swarm/`.

- Pros: one representation everywhere.
- Cons: edits the frozen stack (decision 003), risks the 96 swarm tests, and blocks everyone until done.

## Lead's default, unless B objects at the hour-4 sync

Option 2, with a custom `PathScore` that uses each agent's speed and, for ground agents, a path
length from lane B's grid geometry rather than a straight line. Option 1 is the fallback if the
hour-5 stub check shows B is behind. Option 3 is out.

What B should write down after choosing: the decision doc (`docs/decisions/010-agent-adapter.md`),
and one test that a guard and a drone bid for the same verify task and the nearer one wins on time,
not on straight-line distance.
