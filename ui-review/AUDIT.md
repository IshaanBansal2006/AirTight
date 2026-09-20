# Operator console audit (before polish)

Baseline: git HEAD be721d1, built into `before/app_snapshot.html`, shot by `tools/shoot_all.sh`. 40 PNGs in `before/`.
No page errors in any view. Console output is only: a Chrome warning that the three.js `<link rel=preload>` is unused on
every page that is not the Map (the preload is wasted unless Map is opened), and SwiftShader "GPU stall due to
ReadPixels" (harness only). Owner of `pitch/`: Ishaan, lane C (decision 015).

## Cross-cutting

1. Desktop is a phone column. At 1280x800 the app is a roughly 480 px column centred in a peach page; 60% of the
   screen is empty, the chart and the 3D view gain nothing from the width.
2. One colour, several meanings. Navy `#2b3157` is: drones, the asset, gates, camera poles, camera wedges, agent sensor
   discs, the score line, primary buttons and the selected chip. Teal is "good", "Go2", docks, the deployed dot, the
   money icon and the battery bar. Dusty red `--atk` is attack paths, the alarm line, the alarm marker, "bad" pills,
   the "recorded" dot, unseen places and the deny countdown. The Map lead says "Blue is where the fleet patrolled" but
   trails are navy, teal and periwinkle by agent type, and no legend says which is which.
3. Severity dots (`sev-high/med/low`) have no legend anywhere (Map holes, Fleet memory, Coverage).
4. Spacing is not on one scale: section gaps differ between Home (cards touching the hero), Perception and Logs;
   inline `style=` spacing in the template (`margin-top:10px`, `padding:12px 8px 6px`, `font-size:17px`, `14px`).
5. Number formatting: `0 s/h ... was 0 s/h`, `— ago`, `0 s ago` (should read "now"), "1 /h", "pts/$" never defined,
   "phase 0.55 of the charge cycle" is jargon, "63 of 100" is a percentage of intrusions caught in time but is
   presented as a score.
6. Lower-case machine ids as names everywhere: "go2 1", "drone dock nw", "the the asset" (duplicated article in
   Perception agent cards: "patrolling near the the asset").
7. The centre Swarm orb overlaps content: on Map 390 it covers the text of the Round 1 row; on Logs 390 it covers
   "synchronized". The tab bar has no top fade, rows are cut mid-line.
8. Horizontal scrollers (fleet strip, contacts, quick prompts) clip the last card mid-word with no affordance
   ("go...", "M", the prompt row ends in a sliver).
9. Dark theme works, but every colour hard-coded in JS ignores it: asset, gates, poles, wedges (navy on navy, nearly
   invisible in `map_*_dark`), perception agents and discs, `colorFor` avatars.
10. Fonts come from Google Fonts and three.js from cdnjs: the "self-contained" page degrades offline.

## Home ("Tonight")
Purpose: one number for how well the deployed fleet catches intrusions in time, its cost, and how it compares to
other mixes.
- Hero: the key line is long and wraps under the decorative circle; the range bar has no tick labels on the bar, the
  low label "7 · cheapest mix $14/h" and "best mix 67" sit under it in a different order (value first, then value last).
  Chips "worst tactic 50%", "schedule hidden 51%", "was 14 before the fix" are unexplained.
- Cost card: three identical "$" rows that are really three different claims (points, worst case, ratio); "$61k a year
  at 24/7" is computed in the template from cost difference times 8760; "+17 points for $0" next to "63 of 100".
- Tiles: uneven heights, "0 s/h nobody on patrol, was 0 s/h" reads as broken; "4 attack, fix, re-attack rounds" is
  not a stat of tonight.
- Chart "Every fleet mix, scored": y axis has no title or unit (it is % of intrusions detected in time); x tick
  labels collide ("$41 $46$48" overprint); x is not linear-looking because ticks are only at data; the frontier line
  is unlabelled and unexplained; labels "deployed 63", "baseline 14" overlap neighbouring dots; no CI whiskers although
  `ci` is in the payload; no hover or tap; grid line at 100 is drawn in a different colour.
- The conditions paragraph under the chart is raw methodology text ("reduced-order per-look Bernoulli model, truth
  association, engine v0").
- Fleet strip: battery bars tiny (about 14 px) and unreadable, "charging 47%" vs "battery 12%" in one style.
- About 900 px of empty page below the content in the tall shot is only the tall viewport, not a defect.

## Map
What it is: a three.js scene of the site, a model of the yard you orbit. It is not a data plot: there are no axes,
no scale, no time, and nothing encodes a quantity except the radius and opacity of the red entry disc (miss rate,
undocumented). The data in it is two path sets: the recorded trails of the deployed fleet in one episode (the catch
clip), and one worst attack path per red-team round.
Defects seen in `map_*`:
- Perspective camera, FOV 42, at an oblique angle (HOME_VIEW theta -1.15, phi 0.62, r 300): the rectangular yard
  renders as a skewed diamond, north is not up, there is no compass, distances cannot be compared. The same site is
  drawn north-up and flat on Perception, so the two views do not look like the same place.
- On 1280 the yard fills about 55% of the canvas with large empty margins; on 390 it is small and the top third of
  the canvas is empty. The canvas is near square on desktop, 549x521 backing on phone, pixel ratio capped at 1.5
  (soft on a 2x or 3x screen).
- No grid, no scale bar, no dimensions (the yard is 200 x 120 m, stated only in the environment sheet).
- Nothing is labelled: gates, docks, the asset, cameras have no names. The only labels are "R0".."R3" canvas sprites
  (128x64 texture scaled 16x8 world units): blurry, tiny (about 9 px high text), drawn with `depthTest:false` so they
  float over everything, and R1/R2/R3 sit on the same gate and overprint, so only R0 and R3 can be read.
- Fence is 0.4 m thick boxes 2.4 m high: from this distance a 1 px aliased line; corners do not join.
- Gates are 5 x 3.2 x 1.4 navy boxes, the asset a 9 x 7 x 9 navy cube, poles navy cylinders: all the same colour as
  the drone trails. Docks are pale cyan discs with no meaning given.
- Paths are Catmull-Rom tubes with Lambert shading: the spline overshoots at waypoints (the guard and Go2 trails turn
  into zigzag squiggles), lighting makes one colour look like three, drone trails are subsampled to 60 points so
  they are not the real track; three attack tubes to the same gate merge into one blob. No direction arrows, no
  start/end, no time. Drones are lifted to 8 m, which under perspective offsets them from the ground position.
- Camera wedges at 12% navy are just visible in light, invisible in dark.
- The legend is outside the canvas, lists "decoy" although no decoy is drawn in any round here, and omits trails by
  agent type, docks, gates, asset, wedges, and the disc size.
- Controls: the hint says "pinch to zoom, two fingers to pan" on desktop too; no reset button (double-click only, not
  discoverable); no zoom buttons; wheel zoom hijacks page scroll; no keyboard access; the canvas has no aria label.
- The chip row mixes a radio group (All, Round 0..3) with two toggles (Fleet trails, Attack paths) in one row with the
  same shape; the toggles wrap to a second line on both widths. "Round 1" and "Round 2" look identical to "Round 3"
  (same gate), so the filter seems to do nothing.
- Scene background equals the card surface, so the canvas edge is invisible in light; ground plane is +40 m, arbitrary.
- Theme is read once at scene build; switching theme later leaves the old scene colours.
- "Holes" list: "95% still missed" with "→ fix: stagger" in one grey run-on sentence; rows are clickable (zoom to
  the hole) with no affordance; after the zoom there is no way back except double-click.

## Swarm
Purpose: the human-in-the-loop page: one pending approval, a chat with tools, shared fleet memory, decision history.
- The proposal countdown starts at page load, not when the tab is opened; by the time a viewer arrives it reads
  "Denied automatically in 8 s" or is already denied. "Deadline in 12 s" and "denied in 8 s" are two clocks.
- Deny button is low contrast (pink on pink) next to a heavy Approve.
- Quick prompts row clips at the right edge; "What the link is for" is marketing copy inside an operator screen.
- Fleet memory and evidence are two lists in one card with no divider title; "0 s ago" six times; "24 s evidence"
  uses the big-number slot for a timestamp; the two evidence rows are identical text.
- The numbered "how it works" card and an empty Decisions card push the page to about 1950 px.
- "Shared decision store not available in this view" and "Offline replies built from tonight's recorded data" are
  system notes shown as UI copy.

## Perception
Purpose: a 2D north-up replay of the catch episode with the intruder track score underneath. There is no miss vs
catch switch in the UI (`pep = catchEp` is fixed), although the miss episode is in the payload.
- Canvas labels are clipped at the top and right edges ("drone 1" cut in half at the top, "gro.. / gu.." cut at the
  right); labels overlap each other ("drone 3" over "go2 1") and are drawn in 20 px+ text across the agents.
- Sensor discs (10% navy) cover most of the yard and hide the grid; discs, wedges, asset, gates and agents are all
  navy; agent types are not distinguishable; the intruder is not distinguishable at t = 1 s.
- Grid is every 20 m with no labels, no scale bar, no north mark; grid and padding assume a 200 x 120 site
  (`for x<=200`, `for y<=120`, `W/200`, aspect `120/200`).
- No play, pause or scrub: the bar is display only, playback speed 2.5x is not stated, it loops without saying so.
- Score chart: y axis shows only "23" and "-2", no title, no unit (log-likelihood ratio), no ticks; x axis shows "0 s"
  and nothing at the end; the vertical red line (alarm time) is unlabelled while the dashed horizontal "alarm at 4.0"
  is the threshold: two "alarm" marks; "deadline 34 s" label collides with the axis; no playhead tied to the replay.
- Agent cards re-render every 4 s of replay time (text jumps); odd count leaves a hole in the grid; green text for
  "battery 100% · 0 looks on the track" means nothing; "airborne" vs "battery 100%" inconsistent for drones.
- Coverage list: "— ago", "0 s ago", only 6 of the 8 places fit the rule used in Fleet memory (two lists, two row
  counts, same data).

## Logs ("Record")
Purpose: contacts for the fleet plus an expandable record of alerts, decisions, LLM calls, red-team rounds, archives.
- Tab says "Logs", page title says "Record", and the second heading says "Record" again.
- Contacts are placeholders and the page says so; "Ping" only shows a toast. Cards clip at the right edge.
- Tag chips "ALERT" overflow their 64 px box (text touches both edges); outcome pills "good / bad / noted" are
  lower-case judgement words with no definition.
- Filter chips wrap to two rows on phone ("Archive" alone on row two).
- Expanded entry is a wall of text; seed and fleet id are exposed raw; archive times use the viewer's locale.

## Data sources (from `pitch/build_app.py`)
| Payload key | Source | Read through |
|---|---|---|
| `report` | `pitch/report.json` (`--report`) | contract `Report` |
| `report.fixed`, before/after | `pitch/charts/numbers.json` (`before_after.fixed`) | raw JSON |
| `tokens` | `pitch/charts/token_numbers.json` (optional; `ledger` used in Logs) | raw JSON |
| `episodes.miss/.catch` | `pitch/clips/clips_c.json` names baseline, fixed, tactic_id, seed; logs at `data/clip_logs/{miss,catch}/<fleet>__<tactic>__<seed>.jsonl` | contract `read_episode_log` |
| `site` | `scenarios/logistics_yard/site.json` | contract `Site` |
| `curves` | `scenarios/logistics_yard/sensor_curve.json` (only `fov_deg`, `max_range_m()`) | contract `SensorCurves` |
| `fleet` | `scenarios/logistics_yard/fleets/<fixed>.json` | contract `FleetConfig` |
| `threats` | `--tactics-dir` `top_*.json` (here `results/v3/tactics`; default `data/v3/tactics`) | `SearchResult` |
| `rounds` | `results/minmax/summary.json` and `iter<k>_tactics/top_<family>.json` | raw JSON + `SearchResult` |
| `runs` | `results/*/manifest.json` | raw JSON |
| `built_at` | build clock | |
Runtime only: `window.claude.use('db')` (approvals) and `window.claude.use('sample')` (chat), both absent on file://.
`threats` is in the payload; check it is actually rendered (no use seen in the parts read).

## Hard-coded in the template and parts
- Scenario path `scenarios/logistics_yard` in `build_app.py`; header text "Logistics yard" in the template before JS
  replaces it.
- Perception: site size 200 x 120 (grid loops, scale `W/200`, aspect 120/200), `PAD = 8`, playback 2.5x, redraw 32 ms,
  cards every 4 s, confidence `(score + 2) / 6`, alarm threshold drawn at `Y(4)` with the text "alarm at 4.0" (not
  read from the log or `sim.constants`), y range floor -2 and ceiling 5.
- Sensor fallbacks: range 25 m in `memoryAt`, 40/30/60 m by type and FOV 360 in the replay, 60 deg / 30 m for
  fixed cameras; dock test radius 4 m; `zoneOf` 30 m and "west yard / east yard" split at half width.
- Map: HOME_VIEW, FOV 42, near/far 1/2500, zoom 50..600, pan clamp 120/80, drone height 8 m, all geometry sizes,
  trail subsample 60 points, disc radius `6 + 6*miss`, colours `0x2b3157 0x2e6f6b 0x5c6aa3 0xa07070 0x8fbfbb` and
  ground colours, legend decoy `#bbb092`.
- Swarm: `nowT` falls back to 24 s; countdown and deadline derived but floor at 1; six use-case cards, seven quick
  prompts and the greeting are fixed copy; operator avatar "OP".
- Logs: contacts are invented ("telemetry link, dock nw/se" by index parity, "radio channel 3 (placeholder)"),
  `colorFor` hexes, every "what we would improve" sentence is fixed copy.
- Home: "8-hour shift", hours per year for the "$61k a year" line, severity cut-offs 0.9 / 0.5 on the Map list,
  staleness cut-off 30 s in Fleet memory.
