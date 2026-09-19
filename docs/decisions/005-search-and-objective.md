# Decision 005: Search And Objective

## Context
- The stub episode costs microseconds; the real sim will cost seconds; the sweep at hour 18 is 200 seeds per cell.
- Options considered: random sampling with elites and perturbation, CMA-ES over a flattened parameter vector, a learned adversary as in Ward et al.
- The objective must be comparable across families and must give the search something to climb when the miss rate is flat.

## Decision
Per family: sample 200 valid tactics, keep the 10 best, refine each with 4 perturbations per round for 3 rounds, on the first 20 committed seeds. The adversary score is `0.9 × miss rate + 0.1 × near-miss`, where near-miss is the mean of `clip(1 − margin / response_time, 0, 1)` and margin is seconds between the alarm and the critical detection point.

## Reason
- Random plus elites plus perturbation has no parameter vector to flatten (waypoint counts vary, decoys are optional), needs no gradient, and is explainable in one sentence on a slide; CMA-ES needs a fixed-length encoding and a learned adversary needs training time we do not have.
- The near-miss term is continuous, so two tactics that both missed 3 of 20 are ordered by how close the others came; it is capped at 0.1 so it never outranks an extra miss.
- Every configuration and every before-and-after uses the same seed list, so paired comparisons are exact and the fix loop cannot win by luck.

## Consequences
- `redteam/search.py` writes `top_<family>.json` and `summary.json`; `--inject` folds hand-written and LLM-proposed tactics into the initial population so they compete on equal terms.
- Budget per family is (200 + 10 × 4 × 3) × 20 = 6,400 episodes; on the real sim this is set from B's measured episode time at hour 8 by lowering `n_random` first.
- The stub runner now seeds its RNG from (seed, tactic hash) so tactics differ on the stub; before that fix every tactic scored identically per seed.
