#!/usr/bin/env bash
# Save, apply and inspect patches to the sibling dimOS checkout. See patches/dimos/README.md.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dimos="$root/../dimos"
patches="$root/patches/dimos"
pin="$(awk -F'"' '/^\[dimos\]/{f=1} f&&/^commit/{print $2; exit}' "$root/pins.toml")"
cmd="${1:-status}"
case "$cmd" in
  save)
    name="${2:?usage: dimos_patch.sh save <name>}"
    n=$(ls "$patches"/*.patch 2>/dev/null | wc -l)
    file="$patches/$(printf '%02d' $((n + 1)))-$name.patch"
    if git -C "$dimos" diff --quiet; then echo "no uncommitted changes in $dimos to save" >&2; exit 1; fi
    { echo "# reason: <one line, fill in>"; echo "# base: $pin"; git -C "$dimos" diff; } > "$file"
    echo "saved $file (edit the reason line, then commit it)"
    ;;
  apply)
    actual="$(git -C "$dimos" rev-parse --short HEAD)"
    [[ "$actual" == "$pin"* || "$pin" == "$actual"* ]] || { echo "dimos is at $actual, pin is $pin; check it out first" >&2; exit 1; }
    for f in "$patches"/*.patch; do
      [[ -e "$f" ]] || { echo "no patches"; exit 0; }
      if git -C "$dimos" apply --check "$f" 2>/dev/null; then git -C "$dimos" apply "$f" && echo "applied $(basename "$f")";
      elif git -C "$dimos" apply --reverse --check "$f" 2>/dev/null; then echo "already applied $(basename "$f")";
      else echo "FAILED $(basename "$f"): does not apply cleanly to $actual" >&2; exit 1; fi
    done
    ;;
  status)
    echo "dimos HEAD $(git -C "$dimos" rev-parse --short HEAD), pin $pin"
    for f in "$patches"/*.patch; do
      [[ -e "$f" ]] || { echo "no patches"; exit 0; }
      if git -C "$dimos" apply --reverse --check "$f" 2>/dev/null; then echo "applied   $(basename "$f")"; else echo "not applied $(basename "$f")"; fi
    done
    ;;
  *) echo "usage: dimos_patch.sh save <name> | apply | status" >&2; exit 2 ;;
esac
