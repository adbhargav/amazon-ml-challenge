"""Integer encoding of the normalised records.

All downstream feature computation works on integer arrays:

* token sets (name core / address / street / numbers) as CSR-style
  ``(indptr, indices)`` with vocabulary ids shared between Source 1 and the
  pool, sorted and unique per record, plus one IDF weight per vocabulary id;
* categorical codes (city, state, postcode, house number, unit, name key,
  street key) with 0 meaning "missing";
* small integer / boolean flags.

Everything is stored in one ``.npz`` per split so later stages never need the
strings again except for the rapidfuzz string scorers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl

log = logging.getLogger("ber")

SET_COLS = {"name": "n_core", "addr": "a_tokens", "street": "a_street", "nums": "a_nums"}
CODE_COLS = ["a_city", "a_state", "a_postcode", "a_hn", "a_hn_raw", "a_hn_suffix", "a_unit", "n_key", "a_street_key"]


@dataclass
class Enc:
    """Encoded arrays for one side (S1 or pool); every array has one row per record."""
    n: int = 0
    sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)   # name -> (indptr, indices)
    codes: Dict[str, np.ndarray] = field(default_factory=dict)                      # col -> int32 codes
    hn_int: np.ndarray = None
    is_domain: np.ndarray = None
    script: np.ndarray = None
    oov: np.ndarray = None
    has_alias: np.ndarray = None
    ntok: np.ndarray = None
    a_empty: np.ndarray = None
    landmark: np.ndarray = None
    ncomp: np.ndarray = None
    source: np.ndarray = None
    country: np.ndarray = None           # country code (shared vocabulary)
    nk_cnt_country: np.ndarray = None    # S1 records sharing the name key within the country (log1p)
    nk_cnt_city: np.ndarray = None       # S1 records sharing the name key within the city (log1p)
    nk_cnt_pool_city: np.ndarray = None  # pool records sharing the name key within the city (log1p)


def _sets_to_csr(col: pl.Series, vocab: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    n = col.len()
    df = pl.DataFrame({"rec": np.arange(n, dtype=np.int64), "tok": col}).with_columns(
        pl.col("tok").str.split(" ")
    ).explode("tok").filter(pl.col("tok") != "").unique(["rec", "tok"])
    vdf = pl.DataFrame({"tok": list(vocab.keys()), "vid": np.fromiter(vocab.values(), dtype=np.int32, count=len(vocab))})
    df = df.join(vdf, on="tok", how="inner").sort(["rec", "vid"])
    rec = df["rec"].to_numpy()
    vid = df["vid"].to_numpy().astype(np.int32)
    counts = np.bincount(rec, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, vid


def _vocab_and_idf(cols: list[pl.Series]) -> Tuple[Dict[str, int], np.ndarray]:
    dfs = []
    for i, c in enumerate(cols):
        dfs.append(pl.DataFrame({"rec": np.arange(c.len(), dtype=np.int64) + (i << 40), "tok": c})
                   .with_columns(pl.col("tok").str.split(" ")).explode("tok").filter(pl.col("tok") != "")
                   .unique(["rec", "tok"]).select("tok"))
    df = pl.concat(dfs).group_by("tok").len().sort("tok")
    n_docs = sum(c.len() for c in cols)
    toks = df["tok"].to_list()
    dfreq = df["len"].to_numpy().astype(np.float64)
    idf = (np.log((n_docs + 1.0) / (dfreq + 1.0)) + 1.0).astype(np.float32)
    return {t: i for i, t in enumerate(toks)}, idf


def _codes(s1c: pl.Series, poolc: pl.Series) -> Tuple[np.ndarray, np.ndarray]:
    both = pl.concat([s1c, poolc])
    uniq = both.unique().sort()
    uniq = uniq.filter(uniq != "")
    mapping = pl.DataFrame({"v": uniq, "code": np.arange(1, uniq.len() + 1, dtype=np.int32)})
    coded = pl.DataFrame({"v": both}).join(mapping, on="v", how="left")["code"].fill_null(0).to_numpy().astype(np.int32)
    return coded[: s1c.len()], coded[s1c.len():]


def _hn_int(s: pl.Series) -> np.ndarray:
    return s.map_elements(lambda x: int(x) if x and x.isdigit() and len(x) < 12 else -1, return_dtype=pl.Int64).to_numpy()


def encode(s1n: pl.DataFrame, pooln: pl.DataFrame) -> Tuple[Enc, Enc, Dict[str, np.ndarray]]:
    e1, e2 = Enc(n=s1n.height), Enc(n=pooln.height)
    idfs: Dict[str, np.ndarray] = {}
    for name, col in SET_COLS.items():
        vocab, idf = _vocab_and_idf([s1n[col], pooln[col]])
        idfs[name] = idf
        e1.sets[name] = _sets_to_csr(s1n[col], vocab)
        e2.sets[name] = _sets_to_csr(pooln[col], vocab)
        log.info("encoded %-6s vocab=%d", name, len(vocab))
    for col in CODE_COLS:
        e1.codes[col], e2.codes[col] = _codes(s1n[col], pooln[col])
    e1.country, e2.country = _codes(s1n["country"], pooln["country"])
    for e, df in ((e1, s1n), (e2, pooln)):
        e.hn_int = _hn_int(df["a_hn"])
        e.is_domain = df["n_is_domain"].to_numpy().astype(np.int8)
        e.script = df["n_script"].to_numpy().astype(np.int8)
        e.oov = df["n_oov"].to_numpy().astype(np.int16)
        e.has_alias = df["n_has_alias"].to_numpy().astype(np.int8)
        e.ntok = df["n_ntok"].to_numpy().astype(np.int16)
        e.a_empty = df["a_empty"].to_numpy().astype(np.int8)
        e.landmark = df["a_landmark"].to_numpy().astype(np.int8)
        e.ncomp = df["a_ncomp"].to_numpy().astype(np.int16)
        e.source = df["source"].to_numpy().astype(np.int8)

    # name-key frequency features --------------------------------------------
    s1k = pl.DataFrame({"k": e1.codes["n_key"], "c": e1.country, "city": e1.codes["a_city"]})
    pk = pl.DataFrame({"k": e2.codes["n_key"], "c": e2.country, "city": e2.codes["a_city"]})
    cnt_country = s1k.filter(pl.col("k") > 0).group_by(["k", "c"]).len().rename({"len": "n"})
    cnt_city = s1k.filter((pl.col("k") > 0) & (pl.col("city") > 0)).group_by(["k", "c", "city"]).len().rename({"len": "n"})
    cnt_pool_city = pk.filter((pl.col("k") > 0) & (pl.col("city") > 0)).group_by(["k", "c", "city"]).len().rename({"len": "n"})
    for e, df in ((e1, s1k), (e2, pk)):
        a = df.join(cnt_country, on=["k", "c"], how="left")["n"].fill_null(0).to_numpy()
        b = df.join(cnt_city, on=["k", "c", "city"], how="left")["n"].fill_null(0).to_numpy()
        c = df.join(cnt_pool_city, on=["k", "c", "city"], how="left")["n"].fill_null(0).to_numpy()
        e.nk_cnt_country = np.log1p(a).astype(np.float32)
        e.nk_cnt_city = np.log1p(b).astype(np.float32)
        e.nk_cnt_pool_city = np.log1p(c).astype(np.float32)
    return e1, e2, idfs


# ------------------------------------------------------------------ persistence
def save_enc(path: Path, e1: Enc, e2: Enc, idfs: Dict[str, np.ndarray]) -> None:
    arrs = {}
    for tag, e in (("s1", e1), ("pool", e2)):
        for k, (ip, ix) in e.sets.items():
            arrs[f"{tag}__set__{k}__indptr"] = ip
            arrs[f"{tag}__set__{k}__indices"] = ix
        for k, v in e.codes.items():
            arrs[f"{tag}__code__{k}"] = v
        for f in fields(Enc):
            if f.name in ("n", "sets", "codes"):
                continue
            arrs[f"{tag}__arr__{f.name}"] = getattr(e, f.name)
        arrs[f"{tag}__n"] = np.asarray(e.n)
    for k, v in idfs.items():
        arrs[f"idf__{k}"] = v
    np.savez(str(path), **arrs)


def load_enc(path: Path) -> Tuple[Enc, Enc, Dict[str, np.ndarray]]:
    z = np.load(str(path))
    out = {}
    idfs = {}
    for tag in ("s1", "pool"):
        e = Enc(n=int(z[f"{tag}__n"]))
        for key in z.files:
            if not key.startswith(tag + "__"):
                continue
            _, kind, *rest = key.split("__")
            if kind == "set":
                name, part = rest
                ip, ix = e.sets.get(name, (None, None))
                if part == "indptr":
                    ip = z[key]
                else:
                    ix = z[key]
                e.sets[name] = (ip, ix)
            elif kind == "code":
                e.codes[rest[0]] = z[key]
            elif kind == "arr":
                setattr(e, rest[0], z[key])
        out[tag] = e
    for key in z.files:
        if key.startswith("idf__"):
            idfs[key[5:]] = z[key]
    return out["s1"], out["pool"], idfs
