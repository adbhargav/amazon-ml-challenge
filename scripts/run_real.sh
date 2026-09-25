#!/usr/bin/env bash
# Full run on the real challenge data, sized for a small machine (6 cores / 8 GB):
#   1. train on the dataset_small/ subsample (scripts/make_train_subsample.py) -> work/models
#   2. --mode test once per country shard (scripts/split_test_by_country.sh); blocking never
#      pairs across countries, so the union of the shard outputs equals one big run
#   3. concatenate shard outputs into output/, validate, package
# usage: [TRAIN_DIR=dataset_state] scripts/run_real.sh <team_name> [extra --set overrides...]
set -euo pipefail
TEAM="${1:?usage: $0 <team_name> [--set key=value ...]}"; shift || true
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"
PIPE="code/business_entity_resolution/src/run_pipeline.py"

# Same blocking settings in train and test (rank / count features must match).
COMMON=(--set n_jobs=5 --set chunk_rows=1000000
        --set k_name=12 --set k_name_addr=12 --set k_addr=6 --set k_reverse=2 --set max_union_per_s1=50
        --set prune_max_train_pairs=8000000 --set gbdt_max_train_pairs=5000000)

mkdir -p work output_shards output

TRAIN_DIR="${TRAIN_DIR:-dataset_small}"
echo "### $(date) train on $TRAIN_DIR"
"$PY" "$PIPE" --mode train --data-dir "$TRAIN_DIR" --work-dir work "${COMMON[@]}" "$@" 2>&1 | tee work/train.log

for C in France US India; do
  [[ -d dataset_shards/$C/test ]] || continue
  W="work_shards/$C"; mkdir -p "$W" "output_shards/$C"
  [[ -e "$W/models" ]] || ln -s "$ROOT/work/models" "$W/models"
  echo "### $(date) test shard $C"
  "$PY" "$PIPE" --mode test --data-dir "dataset_shards/$C" --work-dir "$W" --output-dir "output_shards/$C" \
        "${COMMON[@]}" "$@" 2>&1 | tee "$W/test.log"
done

echo "### $(date) combining shards"
for f in matching_results candidate_pairs; do
  first=1
  for C in France US India; do
    s="output_shards/$C/$f.tsv"; [[ -s "$s" ]] || continue
    if [[ $first == 1 ]]; then cp "$s" "output/$f.tsv"; first=0; else tail -n +2 "$s" >> "output/$f.tsv"; fi
  done
done
wc -l output/*.tsv
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
scripts/package_submission.sh "$TEAM"
echo "### $(date) all done"
