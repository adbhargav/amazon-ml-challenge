"""Candidate generation (blocking).

Runs *per country* (no ground-truth pair crosses a country) and unions four
complementary retrieval paths, each contributing a score / rank feature:

1. ``name``       char-3-gram TF-IDF top-k on the core name (+ alt renderings)
2. ``name_addr``  char-3-gram TF-IDF top-k on "core name | street city"
3. ``addr``       char-3-gram TF-IDF top-k on "house-number street city postcode"
                  (rescues address-only matches: cross-script / domain names)
4. exact keys     (name key, same-or-missing state), (city, street key),
                  (house number, street key), each capped by block size
5. ``reverse``    for every pool record its top-k S1 by the name+addr index —
                  supplies the *competition set* used by the one-to-one step

Sparse top-k products use ``sparse_dot_topn`` so memory stays bounded by the
number of kept pairs, never by |S1| x |pool|.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .io_utils import timer

log = logging.getLogger("ber")

PAIR_SCORE_COLS = ["sc_name", "sc_na", "sc_addr", "sc_rev"]
PAIR_RANK_COLS = ["rk_name", "rk_na", "rk_addr", "rk_rev"]
PAIR_FLAG_COLS = ["ex_name", "ex_street", "ex_hn"]
PAIR_COLS = ["s1", "cand"] + PAIR_SCORE_COLS + PAIR_RANK_COLS + PAIR_FLAG_COLS


def _name_text(df: pl.DataFrame) -> List[str]:
    a = df["n_core"].to_list()
    b = df["n_alt1"].to_list()
    return [(x if not y else f"{x} {y}") for x, y in zip(a, b)]


def _name_addr_text(df: pl.DataFrame) -> List[str]:
    return [f"{n} | {s} {c}".strip() for n, s, c in zip(df["n_core"].to_list(), df["a_street"].to_list(), df["a_city"].to_list())]


def _addr_text(df: pl.DataFrame) -> List[str]:
    out = []
    for hn, s, c, pc in zip(df["a_hn"].to_list(), df["a_street"].to_list(), df["a_city"].to_list(), df["a_postcode"].to_list()):
        t = f"{hn} {s} {c} {pc}".strip()
        out.append(t)
    return out


def _fit_vectorizer(texts: List[str], cfg) -> TfidfVectorizer:
    # n-grams present in more than max(tfidf_max_df * N, 20k) records are dropped: they
    # carry ~no IDF weight but dominate the cost of the sparse product.  The absolute
    # floor keeps small inputs (tests, tiny countries) unpruned.
    n_docs = len(texts)
    max_df = max(int(cfg.tfidf_max_df * n_docs), 20_000) if cfg.tfidf_max_df < 1 else n_docs
    max_df = min(max_df, n_docs)
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=tuple(cfg.ngram_range), min_df=min(cfg.tfidf_min_df, max_df),
        max_df=max_df, max_features=cfg.tfidf_max_features, sublinear_tf=True, dtype=np.float32, lowercase=False,
    )
    vec.fit(texts)
    return vec


def _topk_pairs(Q: sp.csr_matrix, PT: sp.csr_matrix, k: int, cfg, chunk: int = 100_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-k columns of Q @ PT per row. Returns (rows, cols, scores, ranks)."""
    rows, cols, vals, ranks = [], [], [], []
    for start in range(0, Q.shape[0], chunk):
        Qc = Q[start:start + chunk]
        C = sp_matmul_topn(Qc, PT, top_n=k, threshold=cfg.tfidf_threshold, sort=True, n_threads=cfg.n_jobs)
        C = C.tocsr()
        C.sum_duplicates()
        nnz_per_row = np.diff(C.indptr)
        r = np.repeat(np.arange(Qc.shape[0], dtype=np.int64) + start, nnz_per_row)
        rk = np.arange(C.nnz, dtype=np.int64) - np.repeat(C.indptr[:-1], nnz_per_row)
        rows.append(r)
        cols.append(C.indices.astype(np.int64))
        vals.append(C.data.astype(np.float32))
        ranks.append(rk.astype(np.int16))
    if not rows:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int16)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(ranks)


def _sparse_index(q_texts: List[str], p_texts: List[str], k: int, k_rev: int, cfg, tag: str,
                  q_part: np.ndarray | None = None, p_part: np.ndarray | None = None):
    """Top-k retrieval on char-n-gram TF-IDF.

    ``q_part`` / ``p_part`` are partition labels (state codes, "" = missing).  When
    given, every query only searches the pool records of its own partition plus
    the pool records with a missing partition label; queries with a missing
    label search the whole pool.  Returns forward pairs (rows, cols, score,
    rank) and reverse pairs (rows, cols, score, rank), both in (query, pool)
    orientation and in local row/col indices.
    """
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.int16))
    if not q_texts or not p_texts or k <= 0:
        return empty, empty
    vec = _fit_vectorizer(q_texts + p_texts, cfg)
    Q = vec.transform(q_texts).tocsr()
    P = vec.transform(p_texts).tocsr()
    log.info("  [%s] tfidf vocab=%d  Q=%s nnz=%d  P=%s nnz=%d", tag, len(vec.vocabulary_), Q.shape, Q.nnz, P.shape, P.nnz)

    if q_part is None:
        partitions = [(np.arange(Q.shape[0]), np.arange(P.shape[0]))]
    else:
        partitions = []
        p_missing = np.flatnonzero(p_part == "")
        for label in np.unique(q_part):
            qi = np.flatnonzero(q_part == label)
            if label == "":
                pi = np.arange(P.shape[0])
            else:
                pi = np.union1d(np.flatnonzero(p_part == label), p_missing)
            if len(pi):
                partitions.append((qi, pi))
        log.info("  [%s] %d partitions", tag, len(partitions))

    fw, rv = [], []
    for qi, pi in partitions:
        Qp = Q[qi]
        Pp = P[pi]
        r, c, v, rk = _topk_pairs(Qp, Pp.T.tocsr(), k, cfg)
        fw.append((qi[r], pi[c], v, rk))
        if k_rev > 0:
            r2, c2, v2, rk2 = _topk_pairs(Pp, Qp.T.tocsr(), k_rev, cfg)
            rv.append((qi[c2], pi[r2], v2, rk2))

    def cat(parts):
        if not parts:
            return empty
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
    return cat(fw), cat(rv)


def _exact_key_pairs(s1c: pl.DataFrame, poolc: pl.DataFrame, keys: List[str], cap: int,
                     state_gate: bool = False) -> pl.DataFrame:
    """Join S1 and pool on ``keys`` (all non-empty), skipping blocks larger than ``cap``."""
    cond = pl.all_horizontal([pl.col(k) != "" for k in keys])
    a = s1c.filter(cond).select(["idx"] + keys + (["a_state"] if state_gate else []))
    b = poolc.filter(cond).select(["idx"] + keys + (["a_state"] if state_gate else []))
    if a.height == 0 or b.height == 0:
        return pl.DataFrame({"s1": pl.Series([], dtype=pl.Int64), "cand": pl.Series([], dtype=pl.Int64)})
    sizes = b.group_by(keys).len().filter(pl.col("len") <= cap).select(keys)
    b = b.join(sizes, on=keys, how="inner")
    if state_gate:
        j = a.join(b, on=keys, how="inner", suffix="_p")
        j = j.filter((pl.col("a_state") == pl.col("a_state_p")) | (pl.col("a_state") == "") | (pl.col("a_state_p") == ""))
    else:
        j = a.join(b, on=keys, how="inner", suffix="_p")
    return j.select([pl.col("idx").cast(pl.Int64).alias("s1"), pl.col("idx_p").cast(pl.Int64).alias("cand")])


def _pairs_frame(rows, cols, vals, ranks, sc_col: str, rk_col: str) -> pl.DataFrame:
    return pl.DataFrame({
        "s1": rows.astype(np.int64), "cand": cols.astype(np.int64),
        sc_col: vals.astype(np.float32), rk_col: ranks.astype(np.int16),
    })


def block_country(s1c: pl.DataFrame, poolc: pl.DataFrame, cfg, country: str) -> pl.DataFrame:
    """``s1c`` / ``poolc`` are the normalised frames restricted to one country
    (with their global ``idx``).  Returns the union pair table with global idx."""
    s1_idx = s1c["idx"].to_numpy().astype(np.int64)
    pool_idx = poolc["idx"].to_numpy().astype(np.int64)
    q_part = s1c["a_state"].to_numpy().astype(object) if cfg.partition_by_state else None
    p_part = poolc["a_state"].to_numpy().astype(object) if cfg.partition_by_state else None
    frames: List[pl.DataFrame] = []

    with timer(f"[{country}] name index"):
        (r, c, v, k), _ = _sparse_index(_name_text(s1c), _name_text(poolc), cfg.k_name, 0, cfg, "name", q_part, p_part)
        frames.append(_pairs_frame(s1_idx[r], pool_idx[c], v, k, "sc_name", "rk_name"))
    with timer(f"[{country}] name+addr index (+reverse)"):
        (r, c, v, k), (rr, rc, rv, rk) = _sparse_index(_name_addr_text(s1c), _name_addr_text(poolc),
                                                       cfg.k_name_addr, cfg.k_reverse, cfg, "name+addr", q_part, p_part)
        frames.append(_pairs_frame(s1_idx[r], pool_idx[c], v, k, "sc_na", "rk_na"))
        if len(rr):
            frames.append(_pairs_frame(s1_idx[rr], pool_idx[rc], rv, rk, "sc_rev", "rk_rev"))
    with timer(f"[{country}] addr index"):
        s1_has = (s1c["a_street"] != "") | (s1c["a_hn"] != "")
        p_has = (poolc["a_street"] != "") | (poolc["a_hn"] != "")
        s1a, pa = s1c.filter(s1_has), poolc.filter(p_has)
        qa = s1a["a_state"].to_numpy().astype(object) if cfg.partition_by_state else None
        pa_part = pa["a_state"].to_numpy().astype(object) if cfg.partition_by_state else None
        (r, c, v, k), _ = _sparse_index(_addr_text(s1a), _addr_text(pa), cfg.k_addr, 0, cfg, "addr", qa, pa_part)
        frames.append(_pairs_frame(s1a["idx"].to_numpy().astype(np.int64)[r],
                                   pa["idx"].to_numpy().astype(np.int64)[c], v, k, "sc_addr", "rk_addr"))
    with timer(f"[{country}] exact keys"):
        ex_name = _exact_key_pairs(s1c, poolc, ["n_key"], cfg.exact_block_cap, state_gate=True).with_columns(pl.lit(1, dtype=pl.Int8).alias("ex_name"))
        ex_street = _exact_key_pairs(s1c, poolc, ["a_city", "a_street_key"], cfg.exact_block_cap).with_columns(pl.lit(1, dtype=pl.Int8).alias("ex_street"))
        ex_hn = _exact_key_pairs(s1c, poolc, ["a_hn", "a_street_key"], cfg.exact_block_cap).with_columns(pl.lit(1, dtype=pl.Int8).alias("ex_hn"))
        frames.extend([ex_name, ex_street, ex_hn])

    with timer(f"[{country}] union"):
        pairs = pl.concat(frames, how="diagonal")
        agg = [pl.col(c).max().fill_null(0.0).cast(pl.Float32) for c in PAIR_SCORE_COLS] + \
              [pl.col(c).min().fill_null(99).cast(pl.Int16) for c in PAIR_RANK_COLS] + \
              [pl.col(c).max().fill_null(0).cast(pl.Int8) for c in PAIR_FLAG_COLS]
        pairs = pairs.group_by(["s1", "cand"]).agg(agg)
        # bound candidates per S1 with a cheap heuristic
        heur = (pl.max_horizontal(["sc_name", "sc_na", "sc_addr", "sc_rev"]) +
                pl.col("ex_name") + pl.col("ex_street") + pl.col("ex_hn"))
        pairs = pairs.with_columns(heur.alias("_h")).sort(["s1", "_h"], descending=[False, True])
        pairs = pairs.with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
        n_before = pairs.height
        pairs = pairs.filter(pl.col("_r") < cfg.max_union_per_s1).drop(["_h", "_r"])
        log.info("  [%s] union: %d pairs (%d after cap), %.1f per S1", country, n_before, pairs.height, pairs.height / max(1, s1c.height))
    return pairs.select(PAIR_COLS)


def block_all(s1n: pl.DataFrame, pooln: pl.DataFrame, cfg) -> pl.DataFrame:
    """Run blocking for every country present in S1."""
    out = []
    for country in s1n["country"].unique().sort().to_list():
        s1c = s1n.filter(pl.col("country") == country)
        poolc = pooln.filter(pl.col("country") == country)
        if poolc.height == 0:
            log.warning("[%s] no pool records for this country; %d S1 get no candidates", country, s1c.height)
            continue
        out.append(block_country(s1c, poolc, cfg, country))
    if not out:
        return pl.DataFrame({c: pl.Series([], dtype=t) for c, t in zip(PAIR_COLS, [pl.Int64, pl.Int64] + [pl.Float32] * 4 + [pl.Int16] * 4 + [pl.Int8] * 3)})
    return pl.concat(out, how="vertical").sort(["s1", "cand"])


def blocking_recall(pairs: pl.DataFrame, gt_pairs: np.ndarray, s1_mask: np.ndarray) -> dict:
    """Fraction of ground-truth pairs (of queried S1) present in ``pairs``."""
    if gt_pairs is None or len(gt_pairs) == 0:
        return {}
    keep = s1_mask[gt_pairs[:, 0]]
    gt = gt_pairs[keep]
    key = gt[:, 0] * (1 << 32) + gt[:, 1]
    pk = pairs["s1"].to_numpy().astype(np.int64) * (1 << 32) + pairs["cand"].to_numpy().astype(np.int64)
    hit = np.isin(key, pk)
    return {"blocking_recall": float(hit.mean()), "n_gt_pairs": int(len(key)), "n_pairs": int(pairs.height),
            "pairs_per_s1": float(pairs.height / max(1, int(s1_mask.sum())))}
