# ui-review tools

Headless-Chrome screenshots of the operator console. Python is the repo venv (needs PIL, already there).
WebGL works headless through SwiftShader (`--use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist`);
three.js r128 comes from cdnjs on the first Map visit, so the machine needs network.

## Rebuild the page

    cd /Users/rishabghosh/Projects/AirTight-ui
    PYTHONPATH=$PWD/src /Users/rishabghosh/Projects/AirTight/.venv/bin/python pitch/build_app.py --tactics-dir results/v3/tactics

## Shoot everything into any folder

    cd /Users/rishabghosh/Projects/AirTight-ui
    ./ui-review/tools/shoot_all.sh ui-review/after

`shoot_all.sh OUT [PAGE]` copies PAGE (default `pitch/app.html`) to `OUT/app_snapshot.html` and shoots the copy, so a
rebuild during the run cannot mix two versions and nothing is written under `pitch/`. It takes about 3.5 minutes
(5 Chromes in parallel). Output: `<page>_<width>_<theme>[_state].png` for home, map, swarm, perception, logs at 1280x800
and 390x844, light and dark, plus states: `map_*_round0`, `map_*_round3`, `map_*_notrails`, `map_*_hole0zoom`,
`home_*_envsheet`, `logs_*_open`, `swarm_*_approved`, and `_full` (tall viewport, whole scroll) for home, swarm, perception.
If a selector used for a state no longer exists the tool prints `no element <selector>` in `errs` and still shoots.

## One shot

    /Users/rishabghosh/Projects/AirTight/.venv/bin/python ui-review/tools/shot.py ui-review/after/app_snapshot.html map 390x844 dark /tmp/x.png --click '#roundSeg button[data-round="0"]'

Options: `--click CSS` (repeatable, clicked in order half way through the wait), `--scroll PX` (scrollTop of `#main`),
`--wait MS` (default 1500, map 4000), `--gl swiftshader|metal|none`.
Each line of output is a JSON measurement: `three` (library loaded), `mapCanvas` backing size, `overflowing` (elements
past the viewport edge; horizontal scrollers are expected there), `errs` (page errors), then any console lines.

## Same camera, before and after

The tool never touches the camera: every map shot is `HOME_VIEW` from `pitch/ui/scene3d.js`
(`theta -1.15, phi 0.62, r 300, target 0,0`, perspective FOV 42), except `hole0zoom` which is what clicking the round 0
row does (`r 120`, target on the entry). If the polish changes HOME_VIEW or the projection, the after shots show the new
default view by design; to compare like with like keep HOME_VIEW, or say so in the after notes.

## How `before/` was made

`pitch/` was already being edited by other agents when QA started, so `before/` is built from git HEAD (be721d1), not
from the working tree: `before/src/` holds `git show HEAD:` copies of the template and the four parts, and

    PYTHONPATH=$PWD/src /Users/rishabghosh/Projects/AirTight/.venv/bin/python pitch/build_app.py --tactics-dir results/v3/tactics --template ui-review/before/src/app_template.html --out ui-review/before/app_snapshot.html
    ./ui-review/tools/shoot_all.sh ui-review/before ui-review/before/app_snapshot.html

Known harness limits: headless Chrome does not exit once WebGL is up, so the tool kills it after the PNG is stable;
the perception replay is shot about 1.5 s in (t = 1 s), it cannot be scrubbed because the page has no scrub control;
Chrome profiles live in `tools/.chrome-profile/` (gitignored).
