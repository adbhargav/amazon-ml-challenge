#!/usr/bin/env python3
"""Fit the normaliser resources (transliteration dictionary, word segmenter, city
lexicon) on the FULL train split, streaming so it fits in a few GB, and save them.

    .venv/bin/python scripts/fit_normalizer_full.py --out work/normalizer_full

Train the models on a subsample with  --set normalizer_dir=work/normalizer_full
so that the dictionary covers every script even when the subsample does not.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from ber.config import Config
from ber.io_utils import read_source_tsv, read_ground_truth
from ber.normalize import Normalizer
from ber.normalize_address import build_city_lexicon
from ber.normalize_name import WordSegmenter, strip_accents
from ber.translit import Translit, has_indic

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dataset/train"); ap.add_argument("--out", default="work/normalizer_full")
a = ap.parse_args(); cfg = Config(); t0 = time.time()

s1 = read_source_tsv(f"{a.split}/train_source1.tsv")
s1_name = dict(zip(s1["entity_id"].to_list(), s1["business_name"].to_list()))
s1_addr = dict(zip(s1["entity_id"].to_list(), s1["business_address"].to_list()))
print(f"S1 loaded: {len(s1_name)} ({time.time()-t0:.0f}s)", flush=True)

indic = {}           # pool id -> (name, address) for records with any Indic text
latin_counts = {}    # for the segmenter: accumulate texts lazily via a generator over files
for k in (2, 3):
    df = read_source_tsv(f"{a.split}/train_source{k}.tsv")
    for eid, n, ad in zip(df["entity_id"].to_list(), df["business_name"].to_list(), df["business_address"].to_list()):
        if has_indic(n) or has_indic(ad):
            indic[eid] = (n, ad)
    del df
print(f"Indic pool records: {len(indic)} ({time.time()-t0:.0f}s)", flush=True)

gt = read_ground_truth(f"{a.split}/train_ground_truth.tsv")
def pairs():
    for sid, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        for pid in ids:
            rec = indic.get(pid)
            if rec is None: continue
            n, ad = rec
            yield s1_name[sid], n
            pa, pb = s1_addr[sid].split(","), ad.split(",")
            if len(pa) == len(pb):
                for x, y in zip(pa, pb):
                    if has_indic(y): yield x.strip(), y.strip()
translit = Translit.fit(pairs(), cfg.translit_min_count, cfg.translit_runner_up_share)
print(f"translit fitted ({time.time()-t0:.0f}s)", flush=True)

def latin_texts():
    for n in s1["business_name"].to_list():
        yield strip_accents(n).lower()
    for k in (2, 3):
        df = read_source_tsv(f"{a.split}/train_source{k}.tsv")
        for n in df["business_name"].to_list():
            if not has_indic(n) and "." not in n:
                yield strip_accents(n).lower()
        del df
segmenter = WordSegmenter.from_texts(latin_texts(), min_count=3)
print(f"segmenter fitted ({time.time()-t0:.0f}s)", flush=True)

city_lex = build_city_lexicon(s1["business_address"].to_list(), s1["country"].to_list(), translit)
print("city lexicon sizes:", {k: len(v) for k, v in city_lex.items()}, flush=True)
Normalizer(translit, segmenter, city_lex).save(Path(a.out))
print(f"saved to {a.out} ({time.time()-t0:.0f}s)", flush=True)
