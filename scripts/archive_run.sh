#!/usr/bin/env bash
# Copy every metric a run produced into results/<name>/ (tracked), leaving the bulky episode logs behind.
#   scripts/archive_run.sh v3 data/v3            # search, sweep report + detail, replays list
#   scripts/archive_run.sh minmax data/minmax
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
name="${1:?usage: archive_run.sh <name> <data_dir>}"
src="${2:?usage: archive_run.sh <name> <data_dir>}"
dst="$root/results/$name"
mkdir -p "$dst"
# metrics: json/jsonl anywhere in the run except the raw episode logs
find "$src" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' \) \
  -not -path '*/search_logs/*' -not -path '*/search_logs_llm/*' -not -path '*/sweep_logs*/*' -not -path '*/logs/*' -not -path '*/replays/*' -not -path '*/llm_cache/*' \
  | while read -r f; do rel="${f#$src/}"; mkdir -p "$dst/$(dirname "$rel")"; cp "$f" "$dst/$rel"; done
# replay logs are small and worth keeping (one full episode per family best)
for d in $(find "$src" -type d -name replays); do rel="${d#$src/}"; mkdir -p "$dst/$rel"; cp "$d"/*.jsonl "$dst/$rel/" 2>/dev/null || true; done
[[ -f "$root/data/llm_calls.jsonl" ]] && cp "$root/data/llm_calls.jsonl" "$dst/llm_calls.jsonl"
{
  echo "{"
  echo "  \"name\": \"$name\","
  echo "  \"archived_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"git_commit\": \"$(git -C "$root" rev-parse --short HEAD)\","
  echo "  \"dimos_commit\": \"$(git -C "$root/../dimos" rev-parse --short HEAD 2>/dev/null || echo unknown)\","
  echo "  \"site_sha256\": \"$(sha256sum "$root/scenarios/logistics_yard/site.json" | cut -c1-12)\","
  echo "  \"curve_sha256\": \"$(sha256sum "$root/scenarios/logistics_yard/sensor_curve.json" | cut -c1-12)\","
  echo "  \"redteam_config_sha256\": \"$(sha256sum "$root/scenarios/logistics_yard/redteam_config.json" | cut -c1-12)\","
  echo "  \"files\": $(cd "$dst" && find . -type f -not -name manifest.json | sort | python3 -c 'import sys,json; print(json.dumps([l.strip()[2:] for l in sys.stdin]))')"
  echo "}"
} > "$dst/manifest.json"
du -sh "$dst" | cut -f1 | xargs -I{} echo "archived $name: {} in $dst"
