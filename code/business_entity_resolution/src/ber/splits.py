"""Validation harness.

Every train S1 receives:

* ``hidden``  – with probability ``hidden_frac`` the S1 is never queried, but its
  S2/S3 records stay in the pool.  This raises the fraction of pool records that
  match *no* queried S1 (distractors) from ~26% to ~37%, which is what the test
  set's higher pool/S1 ratio implies.  Decision parameters are tuned on this
  harder setting.
* ``fold``    – fold id in [0, n_folds) used for GroupKFold out-of-fold (OOF)
  predictions.  Because *all* non-hidden S1 are scored OOF, the whole training
  set is the validation set and there is no separate holdout to keep small.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def assign_splits(n_s1: int, cfg) -> pl.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    hidden = rng.random(n_s1) < cfg.hidden_frac
    fold = rng.integers(0, cfg.n_folds, n_s1).astype(np.int8)
    return pl.DataFrame({
        "idx": np.arange(n_s1, dtype=np.uint32),
        "hidden": hidden,
        "fold": fold,
    })
