#!/usr/bin/env bash
# Verify each sibling checkout in pins.toml is at its pinned commit.
# Exit 1 with the mismatch named so a lane knows which sibling drifted.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
status=0
while IFS='|' read -r name path commit; do
    dir="$root/$path"
    if [[ ! -d "$dir/.git" ]]; then
        echo "MISSING  $name: expected a git checkout at $dir (clone it as a sibling of this repo)"
        status=1; continue
    fi
    actual="$(git -C "$dir" rev-parse --short HEAD)"
    if git -C "$dir" merge-base --is-ancestor "$commit" HEAD 2>/dev/null && [[ "$(git -C "$dir" rev-parse --short "$commit")" == "$actual" ]]; then
        echo "OK       $name @ $actual"
    else
        echo "DRIFT    $name: pinned $commit, checkout at $actual (run: git -C $dir checkout $commit, or update pins.toml)"
        status=1
    fi
done < <(awk -F'"' '
    /^\[/ { gsub(/[\[\]]/, "", $0); name=$0 }
    /^path/ { path=$2 }
    /^commit/ { print name "|" path "|" $2 }
' "$root/pins.toml")
exit $status
