#!/usr/bin/env bash
# Render pitch/deck.md to HTML (always) and PDF (if a Chromium is available) with Marp.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f deck.md ]] || { echo "deck.md missing: run make_charts.py, make_token_chart.py and build_deck.py first" >&2; exit 1; }
npx --yes @marp-team/marp-cli@latest deck.md --no-stdin --html --allow-local-files -o deck.html </dev/null
echo "rendered deck.html"
if npx --yes @marp-team/marp-cli@latest deck.md --no-stdin --html --allow-local-files --pdf -o deck.pdf </dev/null 2>/dev/null; then
    echo "rendered deck.pdf"
else
    echo "deck.pdf skipped: no Chromium found for Marp's PDF export; open deck.html in a browser and print to PDF"
fi
