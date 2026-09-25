#!/usr/bin/env bash
# Build <team_name>_submission.zip in the layout required by the challenge:
#
#   <team>_submission.zip
#   ├── output/matching_results.tsv, output/candidate_pairs.tsv
#   ├── code/business_entity_resolution/{src,README.md,requirements.txt}
#   └── Documentation_template.md
#
# usage: scripts/package_submission.sh <team_name>
set -euo pipefail
TEAM="${1:?usage: $0 <team_name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for f in output/matching_results.tsv output/candidate_pairs.tsv; do
  [[ -s "$f" ]] || { echo "missing $f — run: python code/business_entity_resolution/src/run_pipeline.py --mode test"; exit 1; }
done
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

ZIP="${TEAM}_submission.zip"
rm -f "$ZIP"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp output/matching_results.tsv output/candidate_pairs.tsv "$STAGE/output/"
cp -r code/business_entity_resolution/src code/business_entity_resolution/tests \
      code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt \
      code/business_entity_resolution/pytest.ini "$STAGE/code/business_entity_resolution/"
cp Documentation_template.md "$STAGE/"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
(cd "$STAGE" && zip -qr "$ROOT/$ZIP" .)
rm -rf "$STAGE"
echo "wrote $ZIP"; unzip -l "$ZIP" | tail -n +1 | head -40
