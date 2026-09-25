"""The official metric: macro-averaged F0.5 per Source-1 entity.

For one S1 with predicted set P and true set T:

    tp = |P ∩ T|
    F0.5 = 1.25 * tp / (0.25 * |T| + |P|)        (0 when tp == 0)
    F0.5 = 1.0 when both P and T are empty (correct singleton)
    F0.5 = 0.0 when T is empty and P is not (false merge on a singleton)

which is algebraically identical to (1.25 * prec * rec) / (0.25 * prec + rec).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Set, Tuple

import numpy as np


def f05_single(pred: Set[str] | Set[int], true: Set[str] | Set[int]) -> float:
    if not pred and not true:
        return 1.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    return 1.25 * tp / (0.25 * len(true) + len(pred))


def macro_f05(pred: Mapping, truth: Mapping, ids: Iterable) -> float:
    ids = list(ids)
    if not ids:
        return float("nan")
    s = 0.0
    for i in ids:
        s += f05_single(set(pred.get(i, ())), set(truth.get(i, ())))
    return s / len(ids)


def macro_f05_from_pairs(
    pred_pairs: np.ndarray,
    true_pairs: np.ndarray,
    s1_ids: np.ndarray,
) -> Dict[str, float]:
    """Vectorised scorer on integer pair arrays (n, 2) = (s1_idx, pool_idx).

    Returns a dict with the macro F0.5 plus a breakdown into singleton /
    matched entities, micro precision and recall, and counts.
    """
    s1_ids = np.asarray(s1_ids, dtype=np.int64)
    n = len(s1_ids)
    idx_of = {int(v): i for i, v in enumerate(s1_ids)}

    n_true = np.zeros(n, dtype=np.int64)
    n_pred = np.zeros(n, dtype=np.int64)
    tp = np.zeros(n, dtype=np.int64)

    true_set = set()
    for a, b in true_pairs:
        i = idx_of.get(int(a))
        if i is None:
            continue
        n_true[i] += 1
        true_set.add((int(a), int(b)))
    for a, b in pred_pairs:
        i = idx_of.get(int(a))
        if i is None:
            continue
        n_pred[i] += 1
        if (int(a), int(b)) in true_set:
            tp[i] += 1

    f = np.zeros(n, dtype=np.float64)
    both_empty = (n_true == 0) & (n_pred == 0)
    f[both_empty] = 1.0
    nz = tp > 0
    f[nz] = 1.25 * tp[nz] / (0.25 * n_true[nz] + n_pred[nz])

    single = n_true == 0
    out = {
        "macro_f05": float(f.mean()) if n else float("nan"),
        "n_s1": int(n),
        "f05_singletons": float(f[single].mean()) if single.any() else float("nan"),
        "f05_matched": float(f[~single].mean()) if (~single).any() else float("nan"),
        "n_singletons": int(single.sum()),
        "singleton_rate_true": float(single.mean()) if n else float("nan"),
        "singleton_rate_pred": float((n_pred == 0).mean()) if n else float("nan"),
        "micro_precision": float(tp.sum() / max(1, n_pred.sum())),
        "micro_recall": float(tp.sum() / max(1, n_true.sum())),
        "n_pred_pairs": int(n_pred.sum()),
        "n_true_pairs": int(n_true.sum()),
    }
    return out


def pairs_to_dict(pairs: np.ndarray) -> Dict[int, Set[int]]:
    d: Dict[int, Set[int]] = defaultdict(set)
    for a, b in pairs:
        d[int(a)].add(int(b))
    return d
