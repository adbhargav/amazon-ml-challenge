#!/usr/bin/env python3
"""Score a matching_results.tsv against a ground-truth file with the official metric.

    python src/score.py --pred output/matching_results.tsv --truth dataset/train/train_ground_truth.tsv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.io_utils import read_ground_truth  # noqa: E402
from ber.metrics import macro_f05  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    pred = read_ground_truth(a.pred)          # same 2-column layout
    truth = read_ground_truth(a.truth)
    p = dict(zip(pred["source1_entity_id"].to_list(), pred["matched_entity_ids"].to_list()))
    t = dict(zip(truth["source1_entity_id"].to_list(), truth["matched_entity_ids"].to_list()))
    ids = list(t.keys())
    score = macro_f05(p, t, ids)
    single = [i for i in ids if not t[i]]
    matched = [i for i in ids if t[i]]
    print(f"macro F0.5 = {score:.4f} over {len(ids)} S1  (singletons {macro_f05(p, t, single):.4f} n={len(single)}, "
          f"matched {macro_f05(p, t, matched):.4f} n={len(matched)})")
