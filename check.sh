#!/usr/bin/env bash
# Validate the team submission and print only the scoreboard (no per-submission detail).
# Never prints labels or reports: evaluation/reports/ lists ground-truth transition frames.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
SUB=${1:-evaluation/results/UvAStudents.csv}

$PY evaluation/validate_submission.py "$SUB" | grep -E "^(OK|FAIL)|reports [0-9]+ possessions"
echo
$PY evaluation/run_evaluation.py --no-detail | sed -n '/^scoreboard/,/^Wrote/p' | grep -v '^Wrote'
