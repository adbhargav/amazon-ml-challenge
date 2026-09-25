import numpy as np

from ber.config import Config
from ber.decide import apply_rule, expected_f_select, one_to_one


def test_one_to_one_keeps_argmax_only():
    s1 = np.array([0, 1, 2, 0])
    cand = np.array([5, 5, 5, 6])
    p = np.array([0.9, 0.95, 0.1, 0.8])
    keep = one_to_one(s1, cand, p)
    assert keep.tolist() == [False, True, False, True]
    keep = one_to_one(s1, cand, p, margin=0.1)     # 0.95 - 0.9 < 0.1 -> drop both
    assert keep.tolist() == [False, False, False, True]


def test_expected_f_prefers_empty_for_low_probabilities():
    s1 = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.03, 0.02, 0.95, 0.9])
    keep = expected_f_select(s1, p, n_samples=400, max_k=12, seed=1)
    assert keep[:3].sum() == 0            # singleton decision
    assert keep[3:].sum() == 2            # both confident matches kept


def test_expected_f_drops_weak_tail():
    s1 = np.zeros(4, dtype=np.int64)
    p = np.array([0.99, 0.9, 0.35, 0.05])
    keep = expected_f_select(s1, p, n_samples=800, max_k=12, seed=3)
    assert keep.tolist()[:2] == [True, True] and keep[3] == False


def test_apply_rule_cap():
    cfg = Config(max_matches_per_s1=2, expected_f_samples=64)
    s1 = np.zeros(5, dtype=np.int64)
    cand = np.arange(5)
    p = np.array([0.99, 0.98, 0.97, 0.96, 0.95])
    keep = apply_rule(s1, cand, p, {"rule": "threshold", "threshold": 0.5, "one_to_one": True, "a": 1.0, "b": 0.0}, cfg)
    assert keep.sum() == 2
