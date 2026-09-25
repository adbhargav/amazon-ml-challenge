#!/usr/bin/env python3
"""Build a smaller train split that preserves the pool/S1 structure of the full one.

    python3 scripts/make_train_subsample.py --rate 0.14 --out dataset_small

* keeps a deterministic hash-sample of Source-1 entities (rate ``--rate``),
* keeps every Source-2/3 record matched to a kept S1 (from the ground truth),
* keeps the same fraction of the Source-2/3 records that match *no* S1 at all
  (distractors), so the distractor share of the pool is unchanged,
* drops the records matched only to S1 entities that were not kept.
Stdlib only, streams the files once.
"""
import argparse, hashlib, os, sys

def keep(eid: str, rate: float) -> bool:
    h = int.from_bytes(hashlib.blake2b(eid.encode(), digest_size=8).digest(), "little")
    return h / 2**64 < rate

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="dataset/train")
ap.add_argument("--out", default="dataset_small")
ap.add_argument("--rate", type=float, default=0.14)
a = ap.parse_args()
out = os.path.join(a.out, "train"); os.makedirs(out, exist_ok=True)

kept_s1, n_s1 = set(), 0
with open(os.path.join(a.src, "train_source1.tsv"), encoding="utf8") as f, \
     open(os.path.join(out, "train_source1.tsv"), "w", encoding="utf8") as g:
    g.write(next(f))
    for line in f:
        n_s1 += 1
        eid = line.split("\t", 1)[0]
        if keep(eid, a.rate):
            kept_s1.add(eid); g.write(line)
print(f"S1: kept {len(kept_s1)} / {n_s1}", file=sys.stderr)

matched_any, matched_kept, n_gt = set(), set(), 0
with open(os.path.join(a.src, "train_ground_truth.tsv"), encoding="utf8") as f, \
     open(os.path.join(out, "train_ground_truth.tsv"), "w", encoding="utf8") as g:
    g.write(next(f))
    for line in f:
        n_gt += 1
        s1, ids = line.rstrip("\n").split("\t", 1)
        ids = [x for x in ids.split(",") if x]
        matched_any.update(ids)
        if s1 in kept_s1:
            matched_kept.update(ids); g.write(line)
print(f"GT: {n_gt} rows, {len(matched_any)} matched pool ids overall, {len(matched_kept)} for kept S1", file=sys.stderr)

for k in (2, 3):
    n, kept = 0, 0
    with open(os.path.join(a.src, f"train_source{k}.tsv"), encoding="utf8") as f, \
         open(os.path.join(out, f"train_source{k}.tsv"), "w", encoding="utf8") as g:
        g.write(next(f))
        for line in f:
            n += 1
            eid = line.split("\t", 1)[0]
            if eid in matched_kept or (eid not in matched_any and keep(eid, a.rate)):
                kept += 1; g.write(line)
    print(f"source{k}: kept {kept} / {n}", file=sys.stderr)
