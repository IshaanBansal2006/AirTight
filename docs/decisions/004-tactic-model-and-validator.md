# Decision 004: Tactic Model And Validator

## Context
- The adversary needs a representation that the search can mutate, the LLM can emit, the sim can execute, and a slide can draw.
- Ward et al. 2025 use a closed-loop learned adversary that reacts to the fleet; we have 24 hours and a simulator that does not exist until hour 8.
- Four families were named in the plan: charging window, decoy, blind spot, comms cut.

## Decision
A tactic is open-loop: an entry point, a phase of the charge cycle, one speed, a waypoint path ending at the asset, plus an optional decoy and an optional comms event. Families are samplers over that one schema, not different schemas, and `redteam/validate.py` is the single gate every tactic passes, whatever produced it.

## Reason
- One schema means the search, the proposer and hand-written examples are interchangeable inputs; the `origin` field records provenance for the slide.
- Open-loop paths are reproducible and cheap: an episode is fully determined by (site, fleet, tactic, seed), which the paired before-and-after comparison requires.
- A single validator with actionable messages is what makes LLM output safe to accept: anything it says is rejected with a reason, never silently repaired.

## Consequences
- Validator rules: entry on the perimeter within 1.5 m, speed in [0.8, 2.5] m/s, at most 6 waypoints, last waypoint is the asset, every leg stays inside the perimeter (sampled every eighth of a leg), legs at least 2 m, decoy inside bounds with lead in [10, 120] s, comms cut before the intruder reaches the asset, family payload present. Scenario constants live in `RedTeamConfig` and freeze at hour 10.
- Blind-spot paths use `GeometryCoverage`: fixed-sensor fields of view plus a 15 m dock halo. It is a stand-in until lane B's steady-state snapshot; the `CoverageSource` protocol is the seam.
- The conditions line on every slide must say "open-loop adversary with full knowledge of the patrol policy and charge schedule", which is a stronger adversary than a naive intruder and a weaker one than Ward's.
