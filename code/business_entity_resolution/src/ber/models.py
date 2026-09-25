"""LightGBM helpers: grouped out-of-fold training and fold-averaged inference."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np

log = logging.getLogger("ber")


def _subsample(idx: np.ndarray, y: np.ndarray, max_rows: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """Keep every positive and a random subset of negatives so that len <= max_rows."""
    if max_rows is None or len(idx) <= max_rows:
        return idx
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n_neg = max(0, max_rows - len(pos))
    if n_neg < len(neg):
        neg = rng.choice(neg, size=n_neg, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def train_folds(X: np.ndarray, y: np.ndarray, fold: np.ndarray, groups: np.ndarray, cfg, params: dict,
                num_rounds: int, tag: str, feature_names: List[str]) -> List[Optional[lgb.Booster]]:
    """Train one model per fold on the rows whose ``fold`` differs (GroupKFold by S1).

    ``X`` is the (already subsampled) training matrix.  Early stopping uses an
    inner 10% split of the training rows' *S1 groups* — never the held-out fold —
    so out-of-fold predictions made later with ``predict_oof`` are honest.
    Returns a list indexed by fold id (None for folds without training rows).
    """
    rng = np.random.default_rng(cfg.seed)
    uniq = np.unique(groups)
    val_group = rng.random(len(uniq)) < 0.1                 # 10% of S1 groups for early stopping
    is_val_all = val_group[np.searchsorted(uniq, groups)]
    models: List[Optional[lgb.Booster]] = []
    for f in range(cfg.n_folds):
        tr = np.where(fold != f)[0]
        if len(tr) == 0:
            models.append(None)
            continue
        is_val = is_val_all[tr]
        tr_fit, tr_val = tr[~is_val], tr[is_val]
        if len(tr_val) == 0 or y[tr_val].sum() == 0 or y[tr_fit].sum() == 0:
            is_val = rng.random(len(tr)) < 0.1
            tr_fit, tr_val = tr[~is_val], tr[is_val]
        dtr = lgb.Dataset(X[tr_fit], y[tr_fit], feature_name=feature_names, free_raw_data=False)
        dval = lgb.Dataset(X[tr_val], y[tr_val], reference=dtr, free_raw_data=False)
        model = lgb.train(params, dtr, num_boost_round=num_rounds, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(cfg.gbdt_early_stopping, verbose=False), lgb.log_evaluation(0)])
        models.append(model)
        log.info("[%s] fold %d: train=%d (pos=%d) best_iter=%d val_logloss=%.5f", tag, f, len(tr_fit),
                 int(y[tr_fit].sum()), model.best_iteration, model.best_score["valid_0"]["binary_logloss"])
    return models


def subsample_rows(y: np.ndarray, max_rows: Optional[int], seed: int) -> np.ndarray:
    """Row indices keeping every positive and a random subset of negatives (sorted)."""
    idx = np.arange(len(y))
    return _subsample(idx, y, max_rows, np.random.default_rng(seed))


def predict_oof(models: List[Optional[lgb.Booster]], X: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold prediction: rows of fold f are scored by models[f]."""
    out = np.full(len(X), np.nan, dtype=np.float32)
    for f, m in enumerate(models):
        rows = np.where(fold == f)[0]
        if m is None or len(rows) == 0:
            continue
        out[rows] = m.predict(X[rows], num_iteration=m.best_iteration).astype(np.float32)
    if np.isnan(out).any():  # folds without a model: average of the others
        live = [m for m in models if m is not None]
        nan = np.isnan(out)
        out[nan] = predict_models(live, X[nan])
    return out


def predict_models(models: List[Optional[lgb.Booster]], X: np.ndarray, chunk: int = 5_000_000) -> np.ndarray:
    """Fold-averaged prediction (test mode)."""
    live = [m for m in models if m is not None]
    out = np.zeros(len(X), dtype=np.float32)
    for m in live:
        for s in range(0, len(X), chunk):
            out[s:s + chunk] += m.predict(X[s:s + chunk], num_iteration=m.best_iteration)
    return out / max(1, len(live))


def save_models(models: List[Optional[lgb.Booster]], d: Path, tag: str, feature_names: List[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{tag}_fold*.txt"):
        old.unlink()
    for i, m in enumerate(models):
        if m is not None:
            m.save_model(str(d / f"{tag}_fold{i}.txt"), num_iteration=m.best_iteration)
    (d / f"{tag}_features.json").write_text(json.dumps(feature_names))


def load_models(d: Path, tag: str) -> Tuple[List[lgb.Booster], List[str]]:
    models = [lgb.Booster(model_file=str(p)) for p in sorted(d.glob(f"{tag}_fold*.txt"))]
    if not models:
        raise FileNotFoundError(f"no models for tag {tag} in {d}; run --mode train first")
    names = json.loads((d / f"{tag}_features.json").read_text())
    return models, names


def feature_importance(models: List[Optional[lgb.Booster]], names: List[str], top: int = 30) -> List[Tuple[str, float]]:
    imp = np.zeros(len(names))
    live = [m for m in models if m is not None]
    for m in live:
        imp += m.feature_importance(importance_type="gain")
    imp /= max(1, len(live))
    order = np.argsort(-imp)
    return [(names[i], float(imp[i])) for i in order[:top]]
