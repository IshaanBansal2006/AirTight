# AirTight console: design plan (agent B)

Aim: a measuring instrument and a survey sheet. Ink on paper, hairlines, figures that line up, colour only where it carries meaning.

## System
- Colour: six base values (paper, ink, blue, orange, red, grey). Blue family is the fleet, split by type; orange is the intruder and nothing else; red is an alarm and nothing else; grey is benign. Full list in `pitch/ui/tokens.md`. Dark theme is a slate sheet with the same roles, not black with a neon accent.
- Type: one family, Barlow, with Barlow Semi Condensed for headings and figures. Barlow comes from road-sign and plate lettering (DIN-like), so it reads as labelling on equipment; it has tabular lining figures; the condensed cut lets large figures and dense number columns fit a 390 px column. Already loaded, so no new dependency; weights changed to 400/500/600 (700 dropped). Scale 12/13/15/18/24/44/72. No monospace.
- Space: one scale, 4 to 64 px. Radii 2 and 4 px. 1 px hairline borders; no shadows anywhere.
- Layout: phone column (480 px) up to 1279 px (560 px from 700). From 1280 px the tab bar becomes a left rail and each page is a two-column sheet: Home puts the frontier chart largest on the left with the headline number and its conditions beside it; Map gives the scene the full left column at viewport height; Perception puts replay left, agents and coverage right; Swarm puts the conversation left, memory and record right; Logs puts the record left, contacts right.
- Headline: the score sits in a two-cell row with a "Measured under" conditions block (`#condLine` moved into the hero in the template), always visible.
- Motion: only the environment menu fades in (120 ms) on click. Pulse ring and blinking "recorded" dot removed. A global reduced-motion rule turns everything off.
- Focus: one global `:focus-visible` ring in `--focus`; DOM order is header, nav, main, footer so the rail is tabbed before the page.
- Footer slot: `footer.shellfoot > a.powered`, bottom right above the tab bar on phones, bottom-left corner of the rail on wide screens.

## Self-review against the "looks generated" list: what was there and what changed
- Gradient wash on the hero and avatar, plus a decorative translucent circle: removed. The hero is a plain panel with a 3 px ink top rule.
- Identical rounded cards (18 px) with the same soft shadow: removed. Shadows gone; radii 4 px; stat tiles became one ruled table; savings became ruled rows; section headings carry an ink rule so sections are separated by rules, not boxes.
- Orb tab button with glow: removed; five equal tabs, selected one marked by an ink rule.
- Pills everywhere (chips, segments, buttons, prompts, tags): now rectangular; hero chips are plain label and value text.
- Tracked-out all-caps eyebrow and log body labels: now sentence case, no tracking. (Log tags ALERT/RED/ARCH are emitted in caps by the script; left, see gaps.)
- Cream and sand ground with rose accent: replaced by a cool neutral paper; rose (`--atk`) was doing the work of both intruder and danger, now split into orange intruder and red alarm.
- Dark theme navy with teal accent: now slate with the same semantic roles as light.
- Hover on everything: hover only on real controls, as a flat `--surface-2` fill.
- Animations not caused by an action (pulse, blink): removed.
- "$" in a coloured circle as an icon: hidden.
- Deny button was red: red is reserved for alarms, so Deny is a neutral outlined button.
- My own first plan had a dark hero panel to make the number pop; dropped, since size and weight already do that.

## Gaps for the lead (script-owned, not editable by me)
See the report; listed in the hand-off message.

## The 3D view against Ronil's yard map (branch feat/pitch-realistic-yard-map, 1d9ef89)

Screenshot of his build: `ui-review/ronil/map_1280_light.png`.

What his version does better: a sense of place. Asphalt, lane markings, a warehouse and gates say
"logistics yard" in one second, and "blue team / red team" are words judges already know.

What works against a measuring instrument: containers, trees, grass and sky are not in the site
file, and containers and a tall warehouse imply walls and occlusion that the simulation does not
model, so the picture would claim more than the engine does. The perspective camera crops the site,
sprite labels scale with distance and clip, red is spent on the red team although red is reserved
for alarms, and on a desktop it is still a phone column.

Decision: keep the to-scale orthographic scene and take the sense of place from data only. Entries
are drawn by their `kind` from the site file and cut out of the fence line (vehicle gate, pedestrian
gate, service gate, fence gap). The benign routes in the site file become the yard markings: a
marked delivery lane and a staff footpath. The fenced hard-standing is toned apart from the paper
outside it. The asset is a low footprint whose size the legend calls nominal. The legend says in one
line that only what is in the site file is drawn, because the simulation has no walls or occlusion.
"Fleet" and "Red team" are used as words; colours stay with the tokens.
