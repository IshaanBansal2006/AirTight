# Decision 015: Operator Console as a Five-Page Phone App

## Context
- The first console was four tabs of pitch material (score, threats, replay, approvals) with the deck's PNG charts embedded. It read as a report, not as the product a site operator would open on a phone.
- The user asked for a product-shaped app in Strava's idiom (white ground, orange accent, flowing rounded surfaces) with five pages: a high-level overview with the site's score range and cost effectiveness, an interactive 3D vulnerability map, the human-in-the-loop link to the swarm through the LLM, a recorded perception map, and contacts plus a record of every alert and decision.
- Everything shown must come from data the repository already has: the frozen report, the recorded miss and catch episodes, the archived attack-fix-re-attack rounds, the site model and the LLM ledger.

## Decision
`pitch/build_app.py` assembles one payload (site, curves, fleet, report, threats, two episodes with tracks, detections and score series, the rounds with each round's worst tactic path, the LLM ledger and the results archive index) and `pitch/app_template.html` renders five pages from it with no images: SVG for the frontier and the score sparkline, a 2D canvas for perception, three.js for the map. The swarm page talks to the model through the artifact `sample` capability when the page is opened in Claude, with fleet status, memory lookup, dispatch, posture and what-if exposed as tools, and falls back to scripted replies built from the same data anywhere else. Decisions go through the shared `db` store as before; silence still denies.

## Reason
- One payload and no embedded images keeps the page at about 120 KB, so it opens instantly on a phone and every number on it is traceable to a file in the repository.
- Tools over the model, rather than free text, are what make "the operator talks to the whole swarm" honest: the model can only report what the fleet memory holds and can only act through calls the operator sees logged, and dispatches still land as proposals.
- The scripted fallback means the demo never depends on a network or a model consent prompt.
- The 3D map reuses the replay scene: blue tubes are the fixed fleet's recorded trails, red tubes are the worst tactic of each archived round, so "what still lands after n fixes" is drawn from the rounds themselves.

## Consequences
- The archived `results/minmax` run drives the map's rounds; a later run with canonical names and the schedule-blind column replaces it by pointing `--minmax-dir` at its folder.
- Contacts carry placeholder handles until a site roster exists; the page says so.
- The earlier console artifact stays published but is superseded; the new one is a separate artifact.
