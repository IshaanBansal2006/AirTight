#!/bin/zsh
cd /Users/rishabghosh/Projects/AirTight-ui/ui-review/tools/p
cp ../../../pitch/app.html app_snapshot.html
PY=/Users/rishabghosh/Projects/AirTight/.venv/bin/python
name=$1
size=$2
theme=$3
query=$4
shift 4
$PY shot.py app_snapshot.html perception $size $theme shots/$name.png --query "$query" --wait 5000 "$@" 2>&1 | grep -v "GPU stall"
