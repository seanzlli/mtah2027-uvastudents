#!/usr/bin/env bash
# Validate the team submission and print only the scoreboard (no per-submission detail).
# Never prints labels or reports: evaluation/reports/ lists ground-truth transition frames.
set -euo pipefail
cd "$(dirname "$0")"
# Prefer the venv README tells you to create (it has pandas for the scorer); fall back to the
# interpreter on PATH so this works on a clean clone, before `python3 -m venv .venv` is run.
if [ -z "${PY:-}" ]; then
    if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
SUB=${1:-evaluation/results/UvAStudents.csv}

$PY evaluation/validate_submission.py "$SUB" | grep -E "^(OK|FAIL)|reports [0-9]+ possessions"
echo
$PY evaluation/run_evaluation.py --no-detail | sed -n '/^scoreboard/,/^Wrote/p' | grep -v '^Wrote'
