# Cost per hour by agent type (HELD; assumptions table for the cost axis)

All figures are USD per operating hour, fully loaded, amortized over three years of 24/7 availability
(26,280 h). They are defaults chosen on 2026-09-19 so the sweep can run; each row says what would
change it. **Verify each source before a number goes on a slide.**

| Type | $/h | Derivation | Verify |
|---|---|---|---|
| drone | 7.00 | Drone-in-a-box unit with dock at about $25k hardware, plus about $2k/year software and $3k/year maintenance and battery replacement: (25,000 + 3 × 5,000) / 26,280 ≈ 1.5 $/h; add operations and connectivity at about 5.5 $/h for a fleet operator on call. | quote for a current dock-class drone (DJI Dock 2 or equivalent) and a service contract |
| go2 | 9.00 | Quadruped at about $16k with compute and payload about $6k, three-year life: 22,000 / 26,280 ≈ 0.85 $/h; add about 8 $/h operations, spares and connectivity because ground robots need more hands-on maintenance than docked drones. | current Unitree Go2 Edu list price and a maintenance quote |
| guard | 32.00 | Contract security guard billed rate in US metro markets, about 25 to 40 $/h including agency margin, insurance and scheduling overhead over a 17 to 20 $/h wage. | BLS occupational wage data for security guards plus one contract-agency rate card |

The sweep compares configurations at equal or lower cost, so the ratios matter more than the absolute
values. If the guard rate moves to 40 $/h, every robot configuration looks cheaper against the
guard-only and guard-plus baselines; if it moves to 25 $/h the story narrows but does not flip.
