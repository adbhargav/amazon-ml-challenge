#!/usr/bin/env python3
"""Build a train split that keeps *whole states*, so that the local density of
look-alike businesses (same street, similar name) is exactly the real one.

    .venv/bin/python scripts/make_train_subsample_by_state.py --frac 0.15 --out dataset_state

Needs work/states/train_source{1,2,3}.parquet from scripts/extract_states.py.
Kept:  every S1 whose parsed state is in the chosen set (per country, states are
       added in hash order until `frac` of that country's S1 are covered);
       every S2/S3 record matched to a kept S1 (ground truth);
       every S2/S3 record whose parsed state is in the chosen set (distractors at
       true density, including records of S1 that were *not* kept);
       a `frac` hash-sample of the S2/S3 records with no parsed state (they join
       every partition in blocking, so their share is kept proportional).
"""
import argparse, hashlib, os, sys
import polars as pl

def h(s: str) -> float:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") / 2**64

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="dataset/train"); ap.add_argument("--states", default="work/states")
ap.add_argument("--out", default="dataset_state"); ap.add_argument("--frac", type=float, default=0.15)
a = ap.parse_args()
out = os.path.join(a.out, "train"); os.makedirs(out, exist_ok=True)

s1 = pl.read_parquet(f"{a.states}/train_source1.parquet")
chosen = {}
for country in s1["country"].unique().to_list():
    cnt = s1.filter(pl.col("country") == country).group_by("a_state").len().sort("len", descending=True)
    total = cnt["len"].sum(); states = [(h(country + "|" + s), s, n) for s, n in zip(cnt["a_state"], cnt["len"]) if s != ""]
    acc, sel = 0, []
    for _, s, n in sorted(states):
        if acc >= a.frac * total: break
        sel.append(s); acc += n
    chosen[country] = set(sel)
    print(f"{country}: {len(sel)} states -> {acc}/{total} S1 ({acc/total:.1%}): {sorted(sel)}", file=sys.stderr)

keep_s1 = set(s1.filter(pl.struct(["country", "a_state"]).map_elements(lambda r: r["a_state"] in chosen.get(r["country"], set()), return_dtype=pl.Boolean))["entity_id"].to_list())
n_s1 = 0
with open(f"{a.src}/train_source1.tsv", encoding="utf8") as f, open(f"{out}/train_source1.tsv", "w", encoding="utf8") as g:
    g.write(next(f))
    for line in f:
        n_s1 += 1
        if line.split("\t", 1)[0] in keep_s1: g.write(line)
print(f"S1: kept {len(keep_s1)} / {n_s1}", file=sys.stderr)

matched_kept = set()
with open(f"{a.src}/train_ground_truth.tsv", encoding="utf8") as f, open(f"{out}/train_ground_truth.tsv", "w", encoding="utf8") as g:
    g.write(next(f))
    for line in f:
        sid, ids = line.rstrip("\n").split("\t", 1)
        if sid in keep_s1:
            matched_kept.update(x for x in ids.split(",") if x); g.write(line)
print(f"GT: {len(matched_kept)} matched pool ids for kept S1", file=sys.stderr)

for k in (2, 3):
    st = pl.read_parquet(f"{a.states}/train_source{k}.parquet")
    in_state = set(st.filter(pl.struct(["country", "a_state"]).map_elements(lambda r: r["a_state"] in chosen.get(r["country"], set()), return_dtype=pl.Boolean))["entity_id"].to_list())
    no_state = set(st.filter(pl.col("a_state") == "")["entity_id"].to_list())
    n, kept, by = 0, 0, {"matched": 0, "state": 0, "nostate": 0}
    with open(f"{a.src}/train_source{k}.tsv", encoding="utf8") as f, open(f"{out}/train_source{k}.tsv", "w", encoding="utf8") as g:
        g.write(next(f))
        for line in f:
            n += 1; eid = line.split("\t", 1)[0]
            if eid in matched_kept: by["matched"] += 1
            elif eid in in_state: by["state"] += 1
            elif eid in no_state and h(eid) < a.frac: by["nostate"] += 1
            else: continue
            kept += 1; g.write(line)
    print(f"source{k}: kept {kept} / {n}  {by}", file=sys.stderr)
