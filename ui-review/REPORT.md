# UI polish report (branch `ui-polish`, from main bdcec45)

The UI is `pitch/`: a single self-contained page, `pitch/app.html`, built by `pitch/build_app.py`
from `pitch/app_template.html`, the parts in `pitch/ui/` and a JSON payload read from real files.
Owner: Ishaan, lane C (decision 015). The 3D view is a scene of the site, not a data plot.

## Commands

Install: nothing beyond the repo's environment (`uv sync --extra dev`). No Node package, no bundler.

Build:

    cd /Users/rishabghosh/Projects/AirTight-ui
    PYTHONPATH=$PWD/src /Users/rishabghosh/Projects/AirTight/.venv/bin/python pitch/build_app.py --tactics-dir results/v3/tactics

Run: open `pitch/app.html` in a browser (it is one file; three.js and the font come from their CDNs).

    open /Users/rishabghosh/Projects/AirTight-ui/pitch/app.html

Test and screenshots:

    PYTHONPATH=$PWD/src /Users/rishabghosh/Projects/AirTight/.venv/bin/python -m pytest -q tests/test_pitch_scripts.py
    ./ui-review/tools/shoot_all.sh ui-review/after

The build needs the two recorded episode logs under `data/clip_logs/` (gitignored). They were copied
from the proposals worktree, where they were produced with `run_episode` under `AIRTIGHT_ENGINE=v0`
for the fleet, tactic and seed in `pitch/clips/clips_c.json`. No simulation was run for this job.

## What changed, by priority

P1, the 3D view (`pitch/ui/scene3d.js`, `#tab-map`). Orthographic, one unit is one metre, drawn
from the site file: 10 m grid (heavier every 50 m), scale bar, fence with each entry cut out by its
kind (vehicle gate, pedestrian gate, service gate, fence gap), delivery lane and staff footpath from
the benign routes, docks by user, the asset as a low footprint the legend calls nominal. Drones at
12 m with a drop line and their footprint on the ground; Go2, guard and fixed cameras with wedges;
docked drones flat and dimmed. The intruder path is a constant-width line coloured by time, with
markers at the critical detection point and at the alarm, and the too-late ring round the asset.
Labels are screen-space HTML that hide rather than overlap. Damped orbit that cannot flip or go
below ground, zoom limits, Top / Angled / Follow intruder / Fit site, re-fit on resize. A timeline
with a before/after choice drives the scene and is linked both ways with the replay page through
the `airtight:time` event. The legend is built from the same table as the glyphs, and says that
only what is in the site file is drawn because the simulation has no walls or occlusion.
Deep links for like-for-like captures: `?cam=top|angled|follow&t=<seconds>&ep=miss|catch`.

P2, the design system (`pitch/ui/app.css`, `pitch/ui/tokens.md`, `ui-review/DESIGN.md`). Semantic
colours shared by scene, charts and text: fleet blues split by agent type, one warm colour for the
intruder and the red team, grey for benign traffic, red for alarm states only. Barlow with Barlow
Semi Condensed for figures, tabular lining numerals, one type scale, one spacing scale, 2 and 4 px
radii. A desktop shell with a side rail from 1280 px; the phone layout is kept below that. Removed
after the self-review: gradients, all shadows, 18 px rounded cards, the orb tab button, pills,
tracked-out caps labels, pulse and blink animations, the cream and rose palette, hover effects on
things that are not controls, middle-dot metadata strings.

P3, charts and replay (`pitch/ui/home.js`, `pitch/ui/replay.js`). Home: the headline with its
interval beside an always-visible conditions list; cost against caught in time with titled axes and
units, interval whiskers, direct labels with leader lines, the baseline as a diamond, the fix as an
arrow with its delta, a legend and a plain-words caption; points are keyboard focusable. Replay:
play, previous and next event, scrub, speed, space and arrow keys, a score chart with titled axes,
the alarm threshold and the critical detection point labelled in plain words, a plain-language
event list that seeks on click, and a synchronised before-and-after mode.

P4. Glossary dialog (caught in time, operating point, worst attack found, critical detection
point). The approval countdown now starts when Swarm is first opened rather than at page load.
Sentence case, plain separators, log outcomes in the operator's words ("caught in time", "missed",
"hole found"), the guard no longer shows a battery, empty states name the command to run.

P5. "Powered by dimOS" in the interface's type, bottom corner of every view, linking to
dimensionalos.com. WORDS ONLY: see "Needs a decision".

Final pass, one thing removed per view: Home, the range bar that repeated its two labels; Map, the
lead paragraph that repeated the caption; Swarm, "ago" under every number (said once in the lead);
Perception, the "recorded" badge; Logs, the "noted" chip on entries with no outcome.

## Evidence

`ui-review/before/` and `ui-review/after/`: 40 screenshots each, every page at 1280x800 and 390x844,
light and dark, plus states (map rounds, trails off, zoom to a hole, environment sheet, open log
entry, approved proposal, full-scroll captures). All 40 after-shots report no page errors. The 7
tests in `tests/test_pitch_scripts.py` pass; ruff is clean on `pitch/`. Map before and after are NOT
from the same camera: the before view is the old perspective HOME_VIEW, the after view is the new
orthographic default, by design.

## Not done, not verified

- Nothing was exercised by a person in a real browser: orbit feel, wheel and pinch zoom, the play
  loops, keyboard handling, reduced motion and touch were code-reviewed and screenshot-checked only.
  Frame rate was not measured; `window.__sceneStats.bench(60)` in the console gives ms per frame.
- Hovering an agent does not yet highlight its trace in the score chart; clicking an event jumps the
  replay and, through the shared clock, the scene.
- Layouts between 480 and 1279 px were not looked at. Focus order was not audited page by page.
- Contrast was computed for the token pairs before three late hex tweaks (light `--fleet-camera`,
  light `--ink-3`, dark `--fleet-drone`); re-run `ui-review/tools/b/contrast.py` after adding them.
- The exported-frame requirement for the dimOS mark: the footer is inside the page, so any page
  capture shows it; the 3D canvas alone does not carry it.
- On a phone the Map's labels are dense; static labels could be hidden below 700 px.

## Needs a decision

1. The dimOS logo. `../dimos` holds `dimos/manipulation/visualization/viser/assets/dimensional-logo.svg`
   (216 x 36, one pale-blue variant, #B0E1F0) and `docs/assets/dimensional-logo-master-transparent.png`.
   dimOS is Apache 2.0, which grants no trademark rights, and the only variant is unreadable on the
   light theme without recolouring, which is not allowed. So the words ship alone. Ask the
   Dimensional team for permission and for a dark-on-light variant.
2. Ronil's `origin/feat/pitch-realistic-yard-map` (1d9ef89) rewrites the same Map code inside
   `app_template.html`. This branch moved that code to `pitch/ui/scene3d.js`, so whichever lands
   second needs a hand merge. `ui-review/DESIGN.md` compares the two and says what was taken from
   his (the sense of place, from site-file data only) and what was not (containers, trees, sky:
   they imply occlusion the simulation does not model).
3. Payload gaps, for lane C and lane B: no interval for worst attack found, the schedule-hidden
   rate or the fix delta; no confidence level for the intervals; no `adversary_knowledge` field
   (the line is fixed wording); the operating threshold is not in the payload, so the score chart
   uses one named reference constant (4.0); contact handles on Logs are placeholders and say so.
4. The template was split into `pitch/ui/` parts and `build_app.py` now inlines them. The built
   page was byte-identical at that commit (be721d1), but it changes how Ishaan edits the console.
5. This branch is for review; it was not merged and no PR was opened.

## Dependencies

None added. See `ui-review/DEPENDENCIES.md`.
