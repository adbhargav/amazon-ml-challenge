#!/usr/bin/env python3
"""Parse (state, city) for every record of a split with the fitted normaliser, in chunks.

    .venv/bin/python scripts/extract_states.py --split dataset/train --mode train --out work/states

Writes <out>/<mode>_source{1,2,3}.parquet with entity_id, country, a_state, a_city.
Used by scripts/make_train_subsample_by_state.py.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from ber.io_utils import read_source_tsv
from ber.normalize import Normalizer

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dataset/train"); ap.add_argument("--mode", default="train")
ap.add_argument("--out", default="work/states"); ap.add_argument("--models", default="work/models/normalizer")
ap.add_argument("--chunk", type=int, default=250_000); ap.add_argument("--n-jobs", type=int, default=5)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
norm = Normalizer.load(Path(a.models))
for k in (1, 2, 3):
    t0 = time.time()
    df = read_source_tsv(f"{a.split}/{a.mode}_source{k}.tsv")
    if k == 1:
        norm.extend_city_lexicon(df)
    parts = []
    for s in range(0, df.height, a.chunk):
        sub = df[s:s + a.chunk]
        n = norm.transform(sub, a.n_jobs)
        parts.append(n.select(["entity_id", "country", "a_state", "a_city"]))
        print(f"source{k}: {min(s + a.chunk, df.height)}/{df.height} rows, {time.time() - t0:.0f}s", flush=True)
    res = pl.concat(parts)
    res.write_parquet(str(out / f"{a.mode}_source{k}.parquet"))
    print(f"source{k}: state found {float((res['a_state'] != '').mean()):.3f}, city found {float((res['a_city'] != '').mean()):.3f}", flush=True)
