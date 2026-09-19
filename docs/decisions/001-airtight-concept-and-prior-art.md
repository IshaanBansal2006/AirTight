# Decision 001: Airtight Concept And Prior Art

## Context
- A security-posture engine for commercial sites: a site twin (perimeter, entry points, protected asset, docks, blind spots) patrolled by a mixed fleet of drones, an optional ground robot and human guards, all bidding in one auction, with a battery and dock model so coverage continuity is real.
- Three phases: (1) security mapping on dimOS modules and a `security-site` blueprint; (2) a red-team adversary that plans intrusions exploiting charging windows, decoys and coverage gaps, recording detection, time to detect and human decisions required; (3) a min-max sweep of fleet configurations and patrol policies producing a security score, a cost-versus-detection curve for fleet sizing, and replays of failures.
- The buyer is a building owner, insurer or security firm scoring a site before buying any robot, then re-scoring to show improvement.
- Prior art that must be cited, not ignored: ARES AVERT-PS and Sandia PathTrace sell adversary-path simulation on site twins to nuclear plants; Ward et al. 2025 (arXiv 2509.11971) published a learned adversary against auction-allocated robot patrols with a fleet-size sweep.

## Decision
Build airtight as described above, positioning the contribution above prior art as the mixed air-ground-human fleet, charging-aware continuity, human attention as a measured cost, and a pre-purchase commercial buyer; the headline is the cost-versus-detection curve and the human-decisions-per-hour number, not the simulation.

## Reason
- Adversary-path simulation on twins already exists commercially, so novelty must come from the fleet model and the buyer, not the simulator.
- Ward et al. already cover learned adversary versus auction patrol with a fleet sweep; charging, guards and human attention are the axes they do not measure.
- The number a buyer acts on is cost against detection at a stated false-alarm rate, so that is what every slide must show.

## Consequences
- Contracts in `src/airtight/contracts/` encode the site, fleet, tactic, sensor curve, episode log and report; the report carries the operating point, adversary knowledge and calibration conditions on every number.
- Open decision A: headline metric. Options are human decisions per hour (buyer-facing, needs a defensible threshold model) or coverage gap seconds per hour (simpler, adversary-independent, less novel). Undecided; the report schema carries both so the choice can be made from data.
- Open decision B: first adversary. Options are an LLM planner over the same skills (fast to build, cost per episode, hard to reproduce) or a search over entry point, timing and decoy (reproducible, cheap, needs parameterized families). Undecided; lane C's plan builds the search first and uses the LLM as a proposer feeding it, which keeps both alive until hour 10.
