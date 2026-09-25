#!/usr/bin/env bash
# Full reproduction: train (+ validation report) -> test (+ validator) -> submission zip.
#   scripts/run_all.sh <team_name> [extra --set overrides...]
# e.g. scripts/run_all.sh my_team --set n_jobs=32
set -euo pipefail
TEAM="${1:?usage: $0 <team_name> [--set key=value ...]}"; shift || true
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PIPE="code/business_entity_resolution/src/run_pipeline.py"

for f in dataset/train/train_source1.tsv dataset/train/train_source2.tsv dataset/train/train_source3.tsv \
         dataset/train/train_ground_truth.tsv dataset/test/test_source1.tsv dataset/test/test_source2.tsv dataset/test/test_source3.tsv; do
  [[ -s "$f" ]] || { echo "missing $f"; exit 1; }
done

mkdir -p work
python3 "$PIPE" --mode train "$@" 2>&1 | tee work/train.log
python3 "$PIPE" --mode test  "$@" 2>&1 | tee work/test.log
python3 - <<'EOF'
import json; r = json.load(open("work/train/report.json"))
v = r["decision"]["validation"]["all"]
print(f"\nvalidation macro F0.5 = {v['macro_f05']:.4f}  singletons={v['f05_singletons']:.4f} matched={v['f05_matched']:.4f}")
print("blocking recall:", r["blocking"].get("blocking_recall"), " candidate recall:", r["candidates"].get("blocking_recall"))
EOF
scripts/package_submission.sh "$TEAM"
