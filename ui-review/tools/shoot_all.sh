#!/bin/sh
set -u
OUT="${1:?usage: shoot_all.sh OUTPUT_DIR [PAGE]}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${2:-$ROOT/pitch/app.html}"
PY="${PY:-/Users/rishabghosh/Projects/AirTight/.venv/bin/python}"
SHOT="$ROOT/ui-review/tools/shot.py"
mkdir -p "$OUT"
PAGE="$OUT/app_snapshot.html"
if [ "$SRC" != "$PAGE" ]; then cp "$SRC" "$PAGE"; fi
for vp in 1280x800 390x844; do
  w="${vp%x*}"
  for theme in light dark; do
    for view in home map swarm perception logs; do
      "$PY" "$SHOT" "$PAGE" "$view" "$vp" "$theme" "$OUT/${view}_${w}_${theme}.png" &
    done
    wait
  done
  "$PY" "$SHOT" "$PAGE" map "$vp" light "$OUT/map_${w}_light_round0.png" --click '#roundSeg button[data-round="0"]' &
  "$PY" "$SHOT" "$PAGE" map "$vp" light "$OUT/map_${w}_light_round3.png" --click '#roundSeg button[data-round="3"]' &
  "$PY" "$SHOT" "$PAGE" map "$vp" light "$OUT/map_${w}_light_notrails.png" --click '#roundSeg button[data-layer="sec"]' &
  "$PY" "$SHOT" "$PAGE" map "$vp" light "$OUT/map_${w}_light_hole0zoom.png" --click '#holeList .row[data-round="0"]' &
  "$PY" "$SHOT" "$PAGE" home "$vp" light "$OUT/home_${w}_light_envsheet.png" --click '#envBtn' &
  wait
  "$PY" "$SHOT" "$PAGE" logs "$vp" light "$OUT/logs_${w}_light_open.png" --click '#logList details.log summary' &
  "$PY" "$SHOT" "$PAGE" swarm "$vp" light "$OUT/swarm_${w}_light_approved.png" --click '#proposalBox button' &
  "$PY" "$SHOT" "$PAGE" home "${w}x2600" light "$OUT/home_${w}_light_full.png" &
  "$PY" "$SHOT" "$PAGE" swarm "${w}x2600" light "$OUT/swarm_${w}_light_full.png" &
  "$PY" "$SHOT" "$PAGE" perception "${w}x2000" light "$OUT/perception_${w}_light_full.png" &
  wait
done
ls "$OUT"
