"""Context (second-stage) features built from the stage-1 probabilities.

A pairwise score cannot see the *other* candidates.  These features fix the
three failure modes that pairwise scoring leaves behind:

* **Twins** – same name, different address: within-S1 rank/gap features and a
  "there is a better candidate with the same name key" flag.
* **Orphans / distractors** – a pool record that belongs to a *different*
  (possibly unqueried) S1: competition features, i.e. how well the candidate
  scores against its best other S1 (every S2/S3 record matches at most one S1).
* **S2/S3 disagreement with S1 but agreement with each other** – cluster
  agreement: string / house-number similarity between this candidate and the
  S1's other top candidates.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import polars as pl
from numba import njit
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .encode import Enc
from .pair_features import _cpdist, set_features

log = logging.getLogger("ber")


@njit(cache=True)
def _group_stats(order, s1, p, out_rank, out_gap, out_share, out_cnt, out_sum, out_max):
    """``order`` sorts pairs by (s1, -p).  Fills per-pair within-S1 statistics."""
    n = order.shape[0]
    i = 0
    while i < n:
        j = i
        s = 0.0
        cnt = 0
        while j < n and s1[order[j]] == s1[order[i]]:
            s += p[order[j]]
            if p[order[j]] > 0.5:
                cnt += 1
            j += 1
        pmax = p[order[i]]
        for k in range(i, j):
            o = order[k]
            out_rank[o] = k - i
            out_gap[o] = p[o] - pmax
            out_share[o] = p[o] / s if s > 0 else 0.0
            out_cnt[o] = cnt
            out_sum[o] = s
            out_max[o] = pmax
        i = j


@njit(cache=True)
def _top_others(order, s1, k, out_idx):
    """For each pair, indices (into the pair table) of the top-k *other* pairs of the same S1 (-1 = none)."""
    n = order.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n and s1[order[j]] == s1[order[i]]:
            j += 1
        for a in range(i, j):
            o = order[a]
            m = 0
            for b in range(i, j):
                if b == a:
                    continue
                if m >= k:
                    break
                out_idx[o, m] = order[b]
                m += 1
        i = j


def context_features(pairs: pl.DataFrame, p1: np.ndarray, s1n: pl.DataFrame, pooln: pl.DataFrame, e1: Enc, e2: Enc,
                     idfs: Dict[str, np.ndarray], cfg, k_others: int = 3) -> pl.DataFrame:
    s1 = pairs["s1"].to_numpy().astype(np.int64)
    cand = pairs["cand"].to_numpy().astype(np.int64)
    p = p1.astype(np.float64)
    n = len(p)
    f: Dict[str, np.ndarray] = {}
    f["p1"] = p.astype(np.float32)
    f["p1_logit"] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).astype(np.float32)

    # ---- within-S1 statistics ---------------------------------------------
    order = np.lexsort((-p, s1))
    rank = np.zeros(n, np.int32); gap = np.zeros(n); share = np.zeros(n); cnt = np.zeros(n, np.int32); ssum = np.zeros(n); smax = np.zeros(n)
    _group_stats(order, s1, p, rank, gap, share, cnt, ssum, smax)
    f["s1_rank"] = rank.astype(np.float32)
    f["s1_gap_to_best"] = gap.astype(np.float32)
    f["s1_share"] = share.astype(np.float32)
    f["s1_cnt_over05"] = cnt.astype(np.float32)
    f["s1_sum_p"] = ssum.astype(np.float32)
    f["s1_max_p"] = smax.astype(np.float32)
    src = e2.source[cand]
    for s in (2, 3):
        m = src == s
        cnt_src = np.zeros(e1.n, dtype=np.int64)
        np.add.at(cnt_src, s1[m & (p > 0.5)], 1)
        f[f"s1_cnt_over05_src{s}"] = cnt_src[s1].astype(np.float32)

    # ---- competition across S1 (each pool record matches at most one S1) --------
    order_c = np.lexsort((-p, cand))
    c_rank = np.zeros(n, np.int32); c_gap = np.zeros(n); c_share = np.zeros(n); c_cnt = np.zeros(n, np.int32); c_sum = np.zeros(n); c_max = np.zeros(n)
    _group_stats(order_c, cand, p, c_rank, c_gap, c_share, c_cnt, c_sum, c_max)
    # best *other* S1 for this candidate: max if not the argmax, else the second best
    # second-best probability per candidate
    df = pl.DataFrame({"cand": cand, "p": p}).with_row_index("i")
    top2 = df.sort(["cand", "p"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("cand").alias("r"))
    sec = top2.filter(pl.col("r") == 1).select(["cand", pl.col("p").alias("p2")])
    p2 = df.join(sec, on="cand", how="left")["p2"].fill_null(0.0).to_numpy()
    best_other = np.where(c_rank == 0, p2, c_max)
    f["comp_best_other"] = best_other.astype(np.float32)
    f["comp_margin"] = (p - best_other).astype(np.float32)
    f["comp_is_argmax"] = (c_rank == 0).astype(np.float32)
    f["comp_n_s1_over05"] = c_cnt.astype(np.float32)
    f["comp_n_s1"] = df.group_by("cand").len().join(df, on="cand", how="right").sort("i")["len"].to_numpy().astype(np.float32)

    # ---- cluster agreement with the S1's other top candidates ------------------------
    others = np.full((n, k_others), -1, dtype=np.int64)
    _top_others(order, s1, k_others, others)
    agg_max = np.full(n, np.nan, np.float32); agg_wmean = np.full(n, np.nan, np.float32)
    hn_agree = np.full(n, np.nan, np.float32); opp_max = np.full(n, np.nan, np.float32)
    name_agree = np.full(n, np.nan, np.float32)
    valid = others >= 0
    if valid.any():
        rows, cols = np.nonzero(valid)
        a_pair = rows
        b_pair = others[rows, cols]
        ca, cb = cand[a_pair], cand[b_pair]
        sf = set_features(ca, cb, e2, e2, idfs["addr"], "addr", "cc")
        addr_sim = np.nan_to_num(sf["cc_jac"], nan=0.0)
        hn_a, hn_b = e2.codes["a_hn"][ca], e2.codes["a_hn"][cb]
        hn_same = np.where((hn_a > 0) & (hn_b > 0), (hn_a == hn_b).astype(np.float32), np.nan)
        names = np.asarray(pooln["n_core"].to_list(), dtype=object)
        nsim = _cpdist(names[ca], names[cb], fuzz.token_set_ratio, cfg.n_jobs, cfg.chunk_rows) / 100.0
        w = p[b_pair]
        opp = (e2.source[ca] != e2.source[cb])
        # aggregate per pair
        def agg(values, weights, mask=None):
            mx = np.full(n, np.nan, np.float32); wm = np.full(n, np.nan, np.float32)
            v = values if mask is None else np.where(mask, values, np.nan)
            ok = ~np.isnan(v)
            if ok.any():
                dfa = pl.DataFrame({"i": a_pair[ok], "v": v[ok], "w": weights[ok]})
                g = dfa.group_by("i").agg([pl.col("v").max().alias("mx"), ((pl.col("v") * pl.col("w")).sum() / (pl.col("w").sum() + 1e-6)).alias("wm")])
                mx[g["i"].to_numpy()] = g["mx"].to_numpy()
                wm[g["i"].to_numpy()] = g["wm"].to_numpy()
            return mx, wm
        agg_max, agg_wmean = agg(addr_sim, w)
        hn_agree, _ = agg(hn_same, w)
        opp_max, _ = agg(addr_sim, w, opp)
        name_agree, _ = agg(nsim, w)
    f["clu_addr_max"], f["clu_addr_wmean"] = agg_max, agg_wmean
    f["clu_hn_agree"] = hn_agree
    f["clu_addr_opp_max"] = opp_max
    f["clu_name_max"] = name_agree

    # ---- twin flag: better-scoring candidate with the same name key but a different house number -----
    nk = e2.codes["n_key"][cand]
    hn = e2.codes["a_hn"][cand]
    twin = np.zeros(n, np.float32); twin_gap = np.full(n, np.nan, np.float32)
    if valid.any():
        rows, cols = np.nonzero(valid)
        b_pair = others[rows, cols]
        same_key = (nk[rows] == nk[b_pair]) & (nk[rows] > 0)
        diff_hn = (hn[rows] != hn[b_pair]) & (hn[rows] > 0) & (hn[b_pair] > 0)
        better = p[b_pair] > p[rows]
        m = same_key & diff_hn & better
        if m.any():
            dft = pl.DataFrame({"i": rows[m], "g": (p[b_pair[m]] - p[rows[m]])}).group_by("i").agg(pl.col("g").max())
            twin[dft["i"].to_numpy()] = 1.0
            twin_gap[dft["i"].to_numpy()] = dft["g"].to_numpy()
    f["twin_better"] = twin
    f["twin_gap"] = twin_gap
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
