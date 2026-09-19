# Decision 008: Demo Scenario

## Context
- The example yard in `contracts/examples/` is a test fixture; the demo needs a site a buyer recognises.
- The scenario parameters (site geometry, fleet sweep, adversary limits) freeze at the hour-10 sync and every number in the report depends on them.
- Lanes A and B were not yet ready to run, so the parameters had to be chosen without measured difficulty.

## Decision
The demo site is `scenarios/logistics_yard/`: a 190 × 110 m fenced yard with four entries (vehicle gate, pedestrian gate, service gate, rear fence gap), the asset at the rear of the warehouse footprint, two corner drone docks, a ground-robot dock, two fixed cameras facing into the yard, three benign routes, and a 120 s response time. Adversary limits stay at the code defaults, written out in `redteam_config.json`.

## Reason
- A logistics yard is the buyer archetype from decision 001: perimeter, trucks and staff as benign traffic, one high-value cage, and docks that create real charging windows.
- Four entries of different kinds give the decoy and blind-spot families something to exploit; the rear gap is deliberately far from both cameras.
- 120 s response time at a 1 to 2.5 m/s intruder means the critical detection point sits well inside the yard for every entry, so timeliness rather than sighting is what gets scored.

## Consequences
- The hour-10 difficulty check runs on this site; if the mean detection probability is outside 0.6 to 0.9, the levers are sensor range in the curve file, the rear-gap position and the response time, in that order.
- Hand-written tactic examples stay on the example yard; the demo site's tactics come from the search.
- `tests/test_scenarios.py` proves the site validates and every family samples valid tactics on it.

## Amendment 2026-09-19: response time and drone camera

The first v0 difficulty sweep scored mean detection 0.02 at 120 s response time, 0.42 at 30 s. Two
changes bring it to 0.76, mid-band: response time 25 s (the responder is on site: a guard or the Go2
dispatched from its dock, not an off-site patrol car), and a scenario sensor curve
(`scenarios/logistics_yard/sensor_curve.json`) in which the drone camera is a 360-degree ground disc,
matching lane B's 2-D convention; a 70-degree wedge left the drones nearly blind. Measured with the
baseline fleet, 10 random tactics per family, 10 seeds:

| response time | mean Pd |
|---|---|
| 20 s | 0.83 |
| 25 s | 0.76 |
| 30 s | 0.63 |
| 45 s | 0.39 |

25 s is chosen over 20 s to leave room for part 2 of the engine (battery and phase) to lower detection at the charging window without falling out of band.

## Amendment 2, 2026-09-19: the sprint problem

The first full search on v0 found, in every family, a 2.5 m/s sprint from the main gate: 60 m in 24 s
against a 25 s response time leaves a deadline of zero, so no fleet could ever be timely and the
worst case was 100 percent miss for every configuration. Two knobs lane C owns fix it: intruder
speed cap 2.0 m/s (a brisk jog; a running intruder is a stated exclusion) and response time 20 s.
Measured on v0 with the baseline fleet: random tactics 0.91 mean detection, searched worst case
0.80 miss (charging window and blind spot), 0.40 miss (decoy). The searched worst case, not the
random mean, is what the sweep scores, so 0.91 slightly above the band is accepted. The proper fix
is an adversary task time at the asset (decision 010 revisit); lane B has the parameter, and it is
requested for part 2.

## Gate result with the part 2 engine (batteries and phase), 2026-09-19

Baseline fleet, 10 random tactics per family, 10 seeds, speed cap 2.0 m/s, response time 20 s:
charging window 0.40, decoy 0.72, blind spot 0.85, comms cut 0.76, mean 0.68, in band. The same
gate on the staggered two-drone fleet gives 0.73. The charging-window family is now the weak one on
synchronized charging, which is the story the demo tells. Scenario parameters are frozen here.
