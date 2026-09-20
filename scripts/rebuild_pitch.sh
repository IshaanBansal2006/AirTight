#!/usr/bin/env bash
# Rebuild every pitch artifact from a frozen report, in order, with no hand steps.
#   scripts/rebuild_pitch.sh data/v3/report.json data/v3/tactics [fixed_config]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
report="${1:?usage: rebuild_pitch.sh <report.json> <tactics_dir> [fixed_config]}"
tactics="${2:?usage: rebuild_pitch.sh <report.json> <tactics_dir> [fixed_config]}"
fixed="${3:-d3_go2_guard_stagger}"
scen="$root/scenarios/logistics_yard"
py="$root/.venv/bin/python"
cp "$report" "$root/pitch/report.json"
"$py" "$root/pitch/make_charts.py" --report "$root/pitch/report.json" --site "$scen/site.json" --curves "$scen/sensor_curve.json" --tactics-dir "$tactics" --fixed "$fixed"
"$py" "$root/pitch/make_token_chart.py"
"$py" "$root/pitch/make_clips.py" --tactics-dir "$tactics" --fixed "$fixed" || echo "clips: no miss-and-catch seed found; the deck keeps whatever clip facts exist"
"$py" "$root/pitch/build_deck.py"
"$py" "$root/pitch/fill_writeup.py"
"$py" "$root/pitch/build_app.py" --report "$root/pitch/report.json" --tactics-dir "$tactics"
"$root/pitch/render_deck.sh"
echo "pitch rebuilt from $(basename "$report") with tactics from $tactics"
