#!/usr/bin/env bash
# End-to-end run of the WHOLE pipeline on realistic FAKE data (no DiaData needed).
# Usage (from the project folder):  bash tests/run_e2e.sh [python]
set -euo pipefail
PY=${1:-python3}
T=$(mktemp -d)
$PY tests/make_fake_rich.py "$T/raw" >/dev/null
A="--results $T/results --processed $T/processed"
$PY src/explore_data.py --main "$T/raw/maindatabase.csv" --demographics "$T/raw/demographics.csv" --out "$T/results" >/dev/null
$PY src/nights.py --main "$T/raw/maindatabase.csv" --demographics "$T/raw/demographics.csv" $A >/dev/null
$PY src/features.py $A >/dev/null
$PY src/baselines.py --results "$T/results" --n-boot 100 >/dev/null
$PY src/error_analysis.py --results "$T/results" --n-boot 100 >/dev/null
$PY src/figures.py --results "$T/results" >/dev/null
$PY src/robustness.py --main "$T/raw/maindatabase.csv" --demographics "$T/raw/demographics.csv" $A --n-boot 100 >/dev/null
test -s "$T/results/robustness_summary.csv"
ls "$T/results/figures"/fig*.pdf >/dev/null
echo "E2E PIPELINE PASSED ($T)"
