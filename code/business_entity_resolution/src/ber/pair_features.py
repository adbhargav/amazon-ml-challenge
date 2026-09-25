"""Pairwise features.

Two tiers:

* ``cheap_features`` – pure array gathers + numba set overlaps.  Used by the
  prune model that turns the blocking union into ``candidate_pairs.tsv``.
* ``full_features``  – everything above plus rapidfuzz string scorers on the
  strings (token-set / token-sort / partial ratios, Jaro-Winkler, Levenshtein)
  for the name, its alternative renderings, the concatenated form (domains),
  the street and the whole address.  Used by the stage-1 model.

All features are defined for any country; ``country`` itself is never a
feature so that France (unseen in training) is handled like any other.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import polars as pl
from numba import njit
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .encode import Enc

log = logging.getLogger("ber")


# --------------------------------------------------------------------- numba
@njit(cache=True)
def _set_overlap(ia, ib, ipa, ixa, ipb, ixb, idf, out):
    """For each pair: n_a, n_b, n_common, idf_a, idf_b, idf_common, max_idf_unmatched_a, sum_idf_unmatched_b."""
    for p in range(ia.shape[0]):
        a0, a1 = ipa[ia[p]], ipa[ia[p] + 1]
        b0, b1 = ipb[ib[p]], ipb[ib[p] + 1]
        na = a1 - a0
        nb = b1 - b0
        nc = 0
        idfa = 0.0
        idfb = 0.0
        idfc = 0.0
        max_un_a = 0.0
        sum_un_b = 0.0
        i = a0
        j = b0
        while i < a1 and j < b1:
            va = ixa[i]
            vb = ixb[j]
            if va == vb:
                w = idf[va]
                nc += 1
                idfc += w
                idfa += w
                idfb += w
                i += 1
                j += 1
            elif va < vb:
                w = idf[va]
                idfa += w
                if w > max_un_a:
                    max_un_a = w
                i += 1
            else:
                w = idf[vb]
                idfb += w
                sum_un_b += w
                j += 1
        while i < a1:
            w = idf[ixa[i]]
            idfa += w
            if w > max_un_a:
                max_un_a = w
            i += 1
        while j < b1:
            w = idf[ixb[j]]
            idfb += w
            sum_un_b += w
            j += 1
        out[p, 0] = na
        out[p, 1] = nb
        out[p, 2] = nc
        out[p, 3] = idfa
        out[p, 4] = idfb
        out[p, 5] = idfc
        out[p, 6] = max_un_a
        out[p, 7] = sum_un_b


def set_features(ia: np.ndarray, ib: np.ndarray, ea: Enc, eb: Enc, idf: np.ndarray, key: str, prefix: str) -> Dict[str, np.ndarray]:
    ipa, ixa = ea.sets[key]
    ipb, ixb = eb.sets[key]
    out = np.zeros((len(ia), 8), dtype=np.float32)
    if len(ia):
        _set_overlap(ia.astype(np.int64), ib.astype(np.int64), ipa, ixa, ipb, ixb, idf.astype(np.float32), out)
    na, nb, nc, idfa, idfb, idfc, max_un_a, sum_un_b = out.T
    union = na + nb - nc
    idf_union = idfa + idfb - idfc
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = {
            f"{prefix}_jac": np.where(union > 0, nc / np.maximum(union, 1), np.nan),
            f"{prefix}_cont_a": np.where(na > 0, nc / np.maximum(na, 1), np.nan),
            f"{prefix}_cont_b": np.where(nb > 0, nc / np.maximum(nb, 1), np.nan),
            f"{prefix}_idf_jac": np.where(idf_union > 0, idfc / np.maximum(idf_union, 1e-6), np.nan),
            f"{prefix}_idf_cont_a": np.where(idfa > 0, idfc / np.maximum(idfa, 1e-6), np.nan),
            f"{prefix}_max_idf_unmatched_a": np.where(na > 0, max_un_a, np.nan),
            f"{prefix}_sum_idf_unmatched_b": np.where(nb > 0, sum_un_b, np.nan),
            f"{prefix}_n_common": nc,
            f"{prefix}_n_a": na,
            f"{prefix}_n_b": nb,
        }
    return {k: v.astype(np.float32) for k, v in feats.items()}


# ------------------------------------------------------------- code helpers
def _eq_or_missing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 equal, 0 different, NaN when either side is missing (code 0)."""
    out = (a == b).astype(np.float32)
    out[(a == 0) | (b == 0)] = np.nan
    return out


def cheap_features(pairs: pl.DataFrame, e1: Enc, e2: Enc, idfs: Dict[str, np.ndarray]) -> pl.DataFrame:
    ia = pairs["s1"].to_numpy().astype(np.int64)
    ib = pairs["cand"].to_numpy().astype(np.int64)
    f: Dict[str, np.ndarray] = {}
    for c in ["sc_name", "sc_na", "sc_addr", "sc_rev", "rk_name", "rk_na", "rk_addr", "rk_rev", "ex_name", "ex_street", "ex_hn"]:
        f[c] = pairs[c].to_numpy().astype(np.float32)
    # name / address set overlaps
    f.update(set_features(ia, ib, e1, e2, idfs["name"], "name", "nm"))
    f.update(set_features(ia, ib, e1, e2, idfs["addr"], "addr", "ad"))
    c1, c2 = e1.codes, e2.codes
    f["key_eq"] = ((c1["n_key"][ia] == c2["n_key"][ib]) & (c1["n_key"][ia] > 0)).astype(np.float32)
    f["city_eq"] = _eq_or_missing(c1["a_city"][ia], c2["a_city"][ib])
    f["state_eq"] = _eq_or_missing(c1["a_state"][ia], c2["a_state"][ib])
    f["pc_eq"] = _eq_or_missing(c1["a_postcode"][ia], c2["a_postcode"][ib])
    f["hn_eq"] = _eq_or_missing(c1["a_hn"][ia], c2["a_hn"][ib])
    f["hn_present_s1"] = (c1["a_hn"][ia] > 0).astype(np.float32)
    f["hn_present_c"] = (c2["a_hn"][ib] > 0).astype(np.float32)
    f["cand_addr_empty"] = e2.a_empty[ib].astype(np.float32)
    f["cand_source"] = e2.source[ib].astype(np.float32)
    f["cand_script"] = e2.script[ib].astype(np.float32)
    f["cand_is_domain"] = e2.is_domain[ib].astype(np.float32)
    f["nk_cnt_country"] = e1.nk_cnt_country[ia]
    f["nk_cnt_city"] = e1.nk_cnt_city[ia]
    f["nk_cnt_pool_city"] = e1.nk_cnt_pool_city[ia]
    # candidates per S1 in the union
    s1_counts = np.bincount(ia, minlength=e1.n)
    f["n_cands_union"] = np.log1p(s1_counts[ia]).astype(np.float32)
    return pl.DataFrame({k: v.astype(np.float32) for k, v in f.items()})


# ----------------------------------------------------------- string scorers
def _cpdist(a: np.ndarray, b: np.ndarray, scorer, n_jobs: int, chunk: int) -> np.ndarray:
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = process.cpdist(a[s:s + chunk], b[s:s + chunk], scorer=scorer, workers=n_jobs, dtype=np.float32)
    return out


def _max_over_variants(a_vars: List[np.ndarray], b_vars: List[np.ndarray], scorer, n_jobs: int, chunk: int) -> np.ndarray:
    """max scorer(a_i, b_j) over all non-empty variant combinations; NaN when none."""
    best = np.full(len(a_vars[0]), np.nan, dtype=np.float32)
    for av in a_vars:
        for bv in b_vars:
            mask = (av != "") & (bv != "")
            if not mask.any():
                continue
            sc = _cpdist(av[mask], bv[mask], scorer, n_jobs, chunk)
            cur = best[mask]
            best[mask] = np.where(np.isnan(cur), sc, np.maximum(cur, sc))
    return best


def _hn_features(e1: Enc, e2: Enc, ia: np.ndarray, ib: np.ndarray, s1n: pl.DataFrame, pooln: pl.DataFrame, n_jobs: int, chunk: int) -> Dict[str, np.ndarray]:
    h1, h2 = e1.hn_int[ia], e2.hn_int[ib]
    both = (h1 >= 0) & (h2 >= 0)
    f = {}
    f["hn_state"] = (np.where(both, 0, np.where((h1 < 0) & (h2 < 0), 3, np.where(h1 < 0, 2, 1)))).astype(np.float32)
    f["hn_raw_eq"] = _eq_or_missing(e1.codes["a_hn_raw"][ia], e2.codes["a_hn_raw"][ib])
    diff = np.abs(h1 - h2).astype(np.float64)
    f["hn_absdiff_log"] = np.where(both, np.log1p(diff), np.nan).astype(np.float32)
    f["hn_close"] = np.where(both, (diff <= 5).astype(np.float32), np.nan).astype(np.float32)
    s1h = np.asarray(s1n["a_hn"].to_list(), dtype=object)[ia]
    ch = np.asarray(pooln["a_hn"].to_list(), dtype=object)[ib]
    lev = np.full(len(ia), np.nan, dtype=np.float32)
    if both.any():
        lev[both] = _cpdist(s1h[both], ch[both], Levenshtein.normalized_similarity, n_jobs, chunk)
    f["hn_digit_sim"] = lev
    pre = np.full(len(ia), np.nan, dtype=np.float32)
    if both.any():
        a, b = s1h[both], ch[both]
        pre[both] = np.array([1.0 if (x != y and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x))) else 0.0 for x, y in zip(a, b)], dtype=np.float32)
    f["hn_prefix"] = pre
    f["hn_suffix_eq"] = _eq_or_missing(e1.codes["a_hn_suffix"][ia], e2.codes["a_hn_suffix"][ib])
    f["unit_eq"] = _eq_or_missing(e1.codes["a_unit"][ia], e2.codes["a_unit"][ib])
    return f


def full_features(pairs: pl.DataFrame, s1n: pl.DataFrame, pooln: pl.DataFrame, e1: Enc, e2: Enc,
                  idfs: Dict[str, np.ndarray], cfg) -> pl.DataFrame:
    ia = pairs["s1"].to_numpy().astype(np.int64)
    ib = pairs["cand"].to_numpy().astype(np.int64)
    n_jobs, chunk = cfg.n_jobs, cfg.chunk_rows
    cheap = cheap_features(pairs, e1, e2, idfs)
    f: Dict[str, np.ndarray] = {c: cheap[c].to_numpy() for c in cheap.columns}
    if "prune_p" in pairs.columns:
        f["prune_p"] = pairs["prune_p"].to_numpy().astype(np.float32)
        f["prune_rank"] = pairs["prune_rank"].to_numpy().astype(np.float32)

    def col(df: pl.DataFrame, c: str, idx: np.ndarray) -> np.ndarray:
        return np.asarray(df[c].to_list(), dtype=object)[idx]

    s1_core, c_core = col(s1n, "n_core", ia), col(pooln, "n_core", ib)
    s1_full, c_full = col(s1n, "n_full", ia), col(pooln, "n_full", ib)
    s1_alt1, c_alt1, c_alt2 = col(s1n, "n_alt1", ia), col(pooln, "n_alt1", ib), col(pooln, "n_alt2", ib)
    s1_raw, c_raw = col(s1n, "n_raw_core", ia), col(pooln, "n_raw_core", ib)
    s1_cat, c_cat = col(s1n, "n_concat", ia), col(pooln, "n_concat", ib)

    # ---- name string scorers
    f["nm_tsr"] = _cpdist(s1_full, c_full, fuzz.token_set_ratio, n_jobs, chunk)
    f["nm_tsort"] = _cpdist(s1_core, c_core, fuzz.token_sort_ratio, n_jobs, chunk)
    f["nm_partial"] = _cpdist(s1_core, c_core, fuzz.partial_ratio, n_jobs, chunk)
    f["nm_jw"] = _cpdist(s1_core, c_core, JaroWinkler.normalized_similarity, n_jobs, chunk)
    f["nm_lev"] = _cpdist(s1_core, c_core, Levenshtein.normalized_similarity, n_jobs, chunk)
    f["nm_raw_lev"] = _cpdist(s1_raw, c_raw, Levenshtein.normalized_similarity, n_jobs, chunk)
    f["nm_leet_gain"] = f["nm_lev"] - f["nm_raw_lev"]
    f["nm_concat_lev"] = _cpdist(s1_cat, c_cat, Levenshtein.normalized_similarity, n_jobs, chunk)
    f["nm_concat_partial"] = _cpdist(s1_cat, c_cat, fuzz.partial_ratio, n_jobs, chunk)
    f["nm_alt_tsr"] = _max_over_variants([s1_core, s1_alt1], [c_core, c_alt1, c_alt2], fuzz.token_set_ratio, n_jobs, chunk)
    f["nm_alt_jw"] = _max_over_variants([s1_core, s1_alt1], [c_core, c_alt1, c_alt2], JaroWinkler.normalized_similarity, n_jobs, chunk)
    f["nm_best_tsr"] = np.fmax(f["nm_tsr"], f["nm_alt_tsr"])
    f["core_eq"] = (s1_core == c_core).astype(np.float32)
    s1_sfx, c_sfx = col(s1n, "n_suffix", ia), col(pooln, "n_suffix", ib)
    f["suffix_rel"] = np.where((s1_sfx == "") & (c_sfx == ""), 3, np.where((s1_sfx == "") | (c_sfx == ""), 2, np.where(s1_sfx == c_sfx, 0, 1))).astype(np.float32)
    f["cand_oov"] = e2.oov[ib].astype(np.float32)
    f["cand_has_alias"] = e2.has_alias[ib].astype(np.float32)
    f["s1_ntok"] = e1.ntok[ia].astype(np.float32)
    f["cand_ntok"] = e2.ntok[ib].astype(np.float32)
    f["ntok_ratio"] = (np.minimum(f["s1_ntok"], f["cand_ntok"]) / np.maximum(np.maximum(f["s1_ntok"], f["cand_ntok"]), 1)).astype(np.float32)

    # ---- address
    f.update(_hn_features(e1, e2, ia, ib, s1n, pooln, n_jobs, chunk))
    f.update(set_features(ia, ib, e1, e2, idfs["street"], "street", "st"))
    f.update(set_features(ia, ib, e1, e2, idfs["nums"], "nums", "num"))
    s1_st, c_st = col(s1n, "a_street", ia), col(pooln, "a_street", ib)
    both_st = (s1_st != "") & (c_st != "")
    jw = np.full(len(ia), np.nan, dtype=np.float32)
    tsr = np.full(len(ia), np.nan, dtype=np.float32)
    if both_st.any():
        jw[both_st] = _cpdist(s1_st[both_st], c_st[both_st], JaroWinkler.normalized_similarity, n_jobs, chunk)
        tsr[both_st] = _cpdist(s1_st[both_st], c_st[both_st], fuzz.token_set_ratio, n_jobs, chunk)
    f["st_jw"], f["st_tsr"] = jw, tsr
    s1_ad, c_ad = col(s1n, "a_tokens", ia), col(pooln, "a_tokens", ib)
    both_ad = (s1_ad != "") & (c_ad != "")
    atsr = np.full(len(ia), np.nan, dtype=np.float32)
    if both_ad.any():
        atsr[both_ad] = _cpdist(s1_ad[both_ad], c_ad[both_ad], fuzz.token_set_ratio, n_jobs, chunk)
    f["ad_tsr"] = atsr
    s1_city, c_city = col(s1n, "a_city", ia), col(pooln, "a_city", ib)
    both_city = (s1_city != "") & (c_city != "")
    cjw = np.full(len(ia), np.nan, dtype=np.float32)
    if both_city.any():
        cjw[both_city] = _cpdist(s1_city[both_city], c_city[both_city], JaroWinkler.normalized_similarity, n_jobs, chunk)
    f["city_jw"] = cjw
    f["cand_landmark"] = e2.landmark[ib].astype(np.float32)
    f["s1_ncomp"] = e1.ncomp[ia].astype(np.float32)
    f["cand_ncomp"] = e2.ncomp[ib].astype(np.float32)

    # ---- cross
    f["x_name_addr"] = (f["nm_best_tsr"] / 100.0) * np.nan_to_num(f["ad_jac"], nan=0.0)
    f["x_hn_street"] = np.nan_to_num(f["hn_eq"], nan=0.0) * np.nan_to_num(f["st_jac"], nan=0.0)
    f["x_name_hn"] = (f["nm_best_tsr"] / 100.0) * np.nan_to_num(f["hn_eq"], nan=0.5)
    s1_counts = np.bincount(ia, minlength=e1.n)
    f["n_cands"] = s1_counts[ia].astype(np.float32)
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
