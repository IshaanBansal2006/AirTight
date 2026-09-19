# Track write-up draft (HELD until submission)

Fields follow the usual submission form. Numbers in brackets are filled from `pitch/charts/numbers.json` and `token_numbers.json` by hand at the end, never typed from memory.

## Inspiration

Every commercial site that considers robot security guards faces the same question with no way to answer it: how much safer would we be, and which fleet should we buy? Vendors sell robots; nobody sells the number. Adversary-path simulation exists for nuclear plants, and researchers have trained adversaries against robot patrols, but no tool scores a mixed fleet of drones, a ground robot and human guards, with real batteries and a real cost line, before a purchase.

## What it does

Airtight is a security-posture engine. You give it a site twin: perimeter, entry points, the asset, docks, fixed cameras, benign traffic. It runs a mixed fleet against a red team that searches for the intrusion most likely to succeed, exploiting charging windows, decoys, unwatched approaches and link cuts. It reports one curve, detection against cost per hour, and two numbers beside it: the detection probability against the worst tactic at one false alarm per hour, and the human decisions per hour the fleet demands. Then it proposes a fix, staggering the charge schedule, re-attacks the fixed fleet, and re-scores.

## How we built it

- Robot runtime on dimOS: the Go2 walks, patrols and answers skills over MCP in MuJoCo; the allocator, the approval gate and the fleet are dimOS modules composed into one blueprint.
- A headless simulator for the sweep: kinematic agents, per-look Bernoulli sensing from a calibrated curve, a log-likelihood track score, battery clocks and docks, benign traffic with false alarms.
- A red team: four tactic families over one schema, a validator, elite search with a continuous objective, and an LLM that proposes tactics as programs of primitives which the search then attacks. The LLM never plans an episode, which is why the adversary costs 100x less than an LLM planner. In our runs the cheapest model's proposals were outcompeted by the search; the honest claim is that the proposer is a seed, not the adversary.
- A fleet memory that keeps merging under a cut link: a state-based CRDT with property tests for order independence.
- An offline scorer: thresholds swept on logged scores, the operating point set at one false alarm per hour, bootstrap intervals, paired comparisons on the same seeds.
- Every number on every slide is produced by a script from the frozen report.

## Challenges

- Timing is the whole game. With a 25 s response time and a jogging intruder, the deadline for a run from the nearest gate is zero, and no fleet can ever be timely. We found that in the first real search and fixed the scenario, not the adversary.
- Synchronized charging is a real vulnerability. The moment battery clocks existed, the charging-window family became the weak one on the synchronized fleet while other families stayed high; the worst tactic enters at `rear_fence_gap` at phase 1.00 of the charge cycle.
- Keeping four people from stepping on each other in 24 hours: one schema, one seed list, lanes that only touch their own directories, and a merge gate.

## Accomplishments

- A quantified score with its conditions stated, not a demo: timely detection 0.17 to 0.58 and 0.07 to 0.45 against the worst tactic, from d2_go2_guard_sync at 55 dollars per hour to d3_go2_guard_stagger, on 200 shared seeds; human decisions per hour 12.0 to 41.0.
- The fix loop closes: stagger, re-attack, re-score, on the same seeds.
- Everything reproducible from one command per stage.

## What we learned

- Adversarial search is cheap when the LLM only proposes; the search does the attacking.
- Difficulty must be gated: a scenario where the adversary always or never wins says nothing.
- A CRDT is the right shape for fleet memory, and property tests find the bugs a demo never would.

## What's next

- A learned adversary instead of search, as in the multi-robot patrol literature.
- Calibrated sensor curves for every platform, not just the Go2 camera.
- Multiple assets and an adversary task time, so the deadline reflects what the intruder has to do.
- Scoring a real customer's site from a floor plan.
