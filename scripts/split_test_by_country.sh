#!/usr/bin/env bash
# Split dataset/test/*.tsv into one test split per country under dataset_shards/<country>/test/.
# Blocking only ever pairs records within a country, so running --mode test per shard
# gives identical matches with a fraction of the memory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SRC=dataset/test; OUT=dataset_shards
for k in 1 2 3; do
  f="$SRC/test_source$k.tsv"
  hdr="$(head -1 "$f")"
  # write rows to OUT/<country>/test/test_sourceK.tsv (header added once per file)
  awk -F'\t' -v out="$OUT" -v k="$k" -v hdr="$hdr" 'NR==1{next}
    { c=$4; gsub(/[^A-Za-z0-9_-]/,"_",c); d=out "/" c "/test"; f=d "/test_source" k ".tsv";
      if (!(f in seen)) { system("mkdir -p \"" d "\""); print hdr > f; seen[f]=1 }
      print $0 >> f }' "$f"
done
for d in "$OUT"/*/test; do echo "== $d"; wc -l "$d"/*.tsv; done
