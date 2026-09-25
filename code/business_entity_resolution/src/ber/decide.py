"""From calibrated pair probabilities to the final match lists.

1. **One-to-one**: every S2/S3 record matches at most one S1 in the ground
   truth, so a candidate is kept only for its arg-max S1 (optionally with a
   margin over the runner-up).
2. **Per-S1 expected-F0.5 subset**: sort the S1's candidates by probability and
   pick the prefix length k that maximises the Monte-Carlo expectation of the
   per-entity F0.5 under independent Bernoulli(p) truths.  k = 0 scores
   P(no true match) which is the principled singleton decision.  A 2-parameter
   logit adjustment p' = sigmoid(a * logit(p) + b) is grid-searched on the OOF
   predictions to correct the independence assumption.
3. The alternative global threshold rule is tuned the same way; the better
   rule on OOF is shipped (``decision_rule = "auto"``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numba import njit

from .metrics import macro_f05_from_pairs

log = logging.getLogger("ber")


def one_to_one(s1: np.ndarray, cand: np.ndarray, p: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Boolean mask keeping, for every candidate, only its best S1 (by p) with margin."""
    order = np.lexsort((-p, cand))
    cs = cand[order]
    ps = p[order]
    first = np.ones(len(cs), dtype=bool)
    first[1:] = cs[1:] != cs[:-1]
    # runner-up probability per candidate
    second = np.zeros(len(cs))
    is_second = np.zeros(len(cs), dtype=bool)
    is_second[1:] = (cs[1:] == cs[:-1]) & first[:-1]
    grp_start = np.flatnonzero(first)
    grp_id = np.cumsum(first) - 1
    sec_val = np.zeros(len(grp_start))
    sec_val[grp_id[is_second]] = ps[is_second]
    keep_sorted = first & (ps - sec_val[grp_id] >= margin)
    keep = np.zeros(len(p), dtype=bool)
    keep[order[keep_sorted]] = True
    return keep


@njit(cache=True)
def _expected_f_best_k(order, s1, p, n_samples, max_k, max_group, seed, out_keep):
    """For pairs sorted by (s1, -p): keep per S1 the prefix k maximising E[F0.5].

    E[F0.5(top-k)] is estimated by Monte-Carlo: sample the truth of every
    candidate ~ Bernoulli(p), then F0.5(top-k) = 1.25 * tp_k / (0.25 * T + k)
    where T is the sampled number of true candidates and tp_k the sampled
    true ones among the top-k.  k = 0 scores 1 iff T == 0.
    """
    np.random.seed(seed)
    n = order.shape[0]
    ef = np.zeros(max_k + 1)
    truth = np.zeros(max_group, dtype=np.int64)
    i = 0
    while i < n:
        j = i
        while j < n and s1[order[j]] == s1[order[i]]:
            j += 1
        g = j - i
        m = min(g, max_k)
        for k in range(max_k + 1):
            ef[k] = 0.0
        for s in range(n_samples):
            T = 0
            for k in range(g):
                t = 1 if np.random.random() < p[order[i + k]] else 0
                truth[k] = t
                T += t
            if T == 0:
                ef[0] += 1.0
                continue
            tp = 0
            for k in range(m):
                tp += truth[k]
                if tp > 0:
                    ef[k + 1] += 1.25 * tp / (0.25 * T + (k + 1))
        best_k = 0
        best = ef[0]
        for k in range(1, m + 1):
            if ef[k] > best:
                best = ef[k]
                best_k = k
        for k in range(best_k):
            out_keep[order[i + k]] = True
        i = j


def expected_f_select(s1: np.ndarray, p: np.ndarray, n_samples: int, max_k: int, seed: int = 0) -> np.ndarray:
    order = np.lexsort((-p, s1))
    keep = np.zeros(len(p), dtype=np.bool_)
    if len(p):
        max_group = int(np.bincount(s1).max())
        _expected_f_best_k(order, s1.astype(np.int64), p.astype(np.float64), n_samples, max_k, max_group, seed, keep)
    return keep


def _adjust(p: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def apply_rule(s1: np.ndarray, cand: np.ndarray, p: np.ndarray, params: Dict, cfg) -> np.ndarray:
    keep = np.ones(len(p), dtype=bool)
    if params.get("one_to_one", True):
        keep &= one_to_one(s1, cand, p, params.get("margin", 0.0))
    pa = _adjust(p, params.get("a", 1.0), params.get("b", 0.0))
    if params["rule"] == "threshold":
        sel = pa >= params["threshold"]
    else:
        sel = expected_f_select(s1[keep], pa[keep], cfg.expected_f_samples, cfg.max_matches_per_s1, cfg.seed)
        full = np.zeros(len(p), dtype=bool)
        full[np.flatnonzero(keep)[sel]] = True
        sel = full
    keep &= sel
    # hard cap on matches per S1 (ground truth never exceeds 11)
    order = np.lexsort((-p, s1))
    rank = np.zeros(len(p), dtype=np.int64)
    ks = s1[order]
    starts = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    rank[order] = np.arange(len(p)) - np.repeat(starts, np.diff(np.r_[starts, len(p)]))
    keep &= rank < cfg.max_matches_per_s1
    return keep


def tune(s1: np.ndarray, cand: np.ndarray, p: np.ndarray, gt_pairs: np.ndarray, s1_ids: np.ndarray, cfg) -> Tuple[Dict, Dict]:
    """Grid-search decision parameters on OOF predictions. Returns (best_params, all_results)."""
    results = {}
    best, best_score = None, -1.0

    def score(params):
        keep = apply_rule(s1, cand, p, params, cfg)
        pred = np.stack([s1[keep], cand[keep]], axis=1)
        r = macro_f05_from_pairs(pred, gt_pairs, s1_ids)
        return r["macro_f05"], r

    for margin in ([0.0, 0.05, 0.1] if cfg.one_to_one else [None]):
        o2o = margin is not None
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            params = {"rule": "threshold", "threshold": thr, "one_to_one": o2o, "margin": margin or 0.0, "a": 1.0, "b": 0.0}
            sc, r = score(params)
            results[f"thr={thr} o2o={o2o} m={margin}"] = sc
            if sc > best_score:
                best, best_score = params, sc
        if cfg.decision_rule in ("auto", "expected_f"):
            for a in [0.8, 1.0, 1.25, 1.5]:
                for b in [-1.0, -0.5, 0.0, 0.5]:
                    params = {"rule": "expected_f", "one_to_one": o2o, "margin": margin or 0.0, "a": a, "b": b}
                    sc, r = score(params)
                    results[f"ef a={a} b={b} o2o={o2o} m={margin}"] = sc
                    if sc > best_score:
                        best, best_score = params, sc
    if cfg.decision_rule == "threshold":
        best = {"rule": "threshold", "threshold": cfg.threshold, "one_to_one": cfg.one_to_one, "margin": cfg.one_to_one_margin, "a": 1.0, "b": 0.0}
        best_score = score(best)[0]
    best = dict(best, oof_macro_f05=best_score)
    log.info("decision tuning: best=%s", best)
    return best, results


def save_params(params: Dict, path: Path) -> None:
    path.write_text(json.dumps(params, indent=2))


def load_params(path: Path) -> Dict:
    return json.loads(path.read_text())
