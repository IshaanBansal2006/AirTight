# Airtight deck, eight slides (HELD until submission)

Every number is read from `pitch/charts/numbers.json` or `token_numbers.json`, produced by `make_charts.py` and `make_token_chart.py` from the frozen `report.json`. If a number is not in those files, it does not go on a slide.

| # | Slide | Content | Source |
|---|-------|---------|--------|
| 1 | The question a buyer asks | "How secure is this site, and what should I buy?" One sentence on who pays: owners, insurers, security firms, before any robot. | none |
| 2 | Site twin and mixed fleet | Vulnerability map with the site, entry points, docks, sensor fields of view. Drones, a ground robot and guards bid in one auction; batteries and docks make coverage continuity real. | `vulnerability_map.png` |
| 3 | The red team | Four tactic families. Search plus an LLM proposer that only proposes; search does the attacking. Worst tactic per family drawn on the map. | `numbers.json: worst_tactics` |
| 4 | What the adversary found | The charging-window attack on synchronized charging: phase, entry, miss rate. Replay clip A (the miss). | `numbers.json: worst_tactics`, lane A clip |
| 5 | The score | Detection versus cost with intervals; the frontier. The conditions line under the chart, always. | `cost_vs_detection.png` |
| 6 | The fix and the re-attack | Staggered charging: before and after on the same seeds, and the adversary's number after re-attacking the fixed configuration. Replay clip B (the catch). | `before_after.png`, lane A clip |
| 7 | Human attention as a cost | Decisions per hour and coverage gap per hour beside detection; the token chart for why the adversary is affordable. | `before_after.png`, `token_cost.png` |
| 8 | What we would sell and what is next | Score, vulnerability map, re-score after a purchase. Roadmap: learned adversary, calibrated sensors on more platforms, fleet memory under link loss (if C5 shipped, show it here). | none |

Conditions line, verbatim on slides 5 to 7: operating point, seed count, adversary knowledge, sensor calibration method, reduced-order detection model.
