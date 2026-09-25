import numpy as np
import pytest

from ber.metrics import f05_single, macro_f05, macro_f05_from_pairs


def test_official_example():
    # from the problem statement: pred [A,B,C], truth [A,C] -> 0.714
    assert f05_single({"a", "b", "c"}, {"a", "c"}) == pytest.approx(0.714, abs=1e-3)


def test_singleton_rules():
    assert f05_single(set(), set()) == 1.0
    assert f05_single({"x"}, set()) == 0.0
    assert f05_single(set(), {"x"}) == 0.0
    assert f05_single({"x"}, {"x"}) == 1.0


def test_identity_with_precision_recall_form():
    P, T = {1, 2, 3, 4}, {2, 3, 5}
    tp = len(P & T)
    prec, rec = tp / len(P), tp / len(T)
    expected = 1.25 * prec * rec / (0.25 * prec + rec)
    assert f05_single(P, T) == pytest.approx(expected)


def test_macro_and_vectorised_agree():
    truth = {0: {10, 11}, 1: set(), 2: {12}, 3: {13, 14, 15}}
    pred = {0: {10, 11, 16}, 1: set(), 2: {17}, 3: {13}}
    m = macro_f05(pred, truth, [0, 1, 2, 3])
    tp = np.array([[0, 10], [0, 11], [2, 12], [3, 13], [3, 14], [3, 15]])
    pp = np.array([[0, 10], [0, 11], [0, 16], [2, 17], [3, 13]])
    r = macro_f05_from_pairs(pp, tp, np.array([0, 1, 2, 3]))
    assert r["macro_f05"] == pytest.approx(m)
    assert r["n_singletons"] == 1
    assert r["f05_singletons"] == 1.0
