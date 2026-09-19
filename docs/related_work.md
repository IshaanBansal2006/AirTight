# Related work and our claim (HELD until submission)

Three things exist that a judge or buyer will compare us to. Cite them; do not pretend they are absent.

**Commercial adversary-path simulation on site twins.** ARES Security's AVERT-PS and Sandia National
Laboratories' PathTrace model a protected site and search for adversary paths against a fixed
physical protection system: sensors, barriers, response force, response time. Their customers are
nuclear and other critical facilities; their outputs are vulnerability paths and probability of
interruption. They model a static system and a human response force. They do not model a mobile robot
fleet with charging cycles, an auction that decides which agent responds, or the operator's decision
load. *(Verify product names and current scope on the vendors' pages before the write-up; state only
what their public material says.)*

**Learned adversaries against robot patrols.** Ward et al., "Time-Constrained Intelligent Adversaries
for Automation Vulnerability Testing: A Multi-Robot Patrol Case Study", arXiv:2509.11971 (September
2025). A machine-learning adversary observes patrol behaviour and attempts undetected access within a
time limit; it is evaluated against several patrol strategies including a dynamic auction-based
allocation (DTAP), online Bayesian learning (CBLS) and a reckoning baseline (ER), with results across
fleet sizes. This is the closest academic neighbour: adversary versus auction-allocated patrol, with a
fleet sweep. Their fleet is homogeneous robots on a patrol graph; there is no battery or dock model, no
human guards in the allocation, and the reported cost is detection, not operator attention or dollars.

**Our claim, stated so it stays true.** Above those three we add, in one system: (1) a mixed fleet of
drones, a ground robot and human guards bidding in one auction; (2) charging-aware continuity, with
docks and staggered schedules as a design variable the adversary can exploit and the fix loop can
tune; (3) human attention as a measured cost, reported as decisions per hour beside detection and
dollars; (4) a buyer who scores a site before purchasing any robot and re-scores after. The headline
is the cost-versus-detection curve and the decisions-per-hour number, not the simulator.

**What we do not claim.** Our adversary is open-loop with full knowledge of the patrol policy and the
charge schedule (decisions 004 and 005), which is weaker than Ward's closed-loop learner and stronger
than a naive intruder; the detection model is reduced-order (per-look Bernoulli with truth
association) calibrated on one camera; the sweep uses simulated, not measured, fleets.

Citation to paste: Ward et al. (2025). *Time-Constrained Intelligent Adversaries for Automation
Vulnerability Testing: A Multi-Robot Patrol Case Study.* arXiv:2509.11971. https://arxiv.org/abs/2509.11971
