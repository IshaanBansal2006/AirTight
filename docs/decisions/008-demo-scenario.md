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
