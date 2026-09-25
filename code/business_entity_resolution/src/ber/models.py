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


def train_oof(X: np.ndarray, y: np.ndarray, s1_fold: np.ndarray, s1_idx: np.ndarray, cfg, params: dict,
              num_rounds: int, tag: str, feature_names: List[str]) -> Tuple[np.ndarray, List[lgb.Booster]]:
    """GroupKFold by S1 fold id.  Returns (oof_pred, fold_models).

    Early stopping uses an inner 10% split of the *training* S1 (never the
    held-out fold), so the OOF predictions are honest.
    """
    rng = np.random.default_rng(cfg.seed)
    folds = s1_fold[s1_idx]
    oof = np.full(len(y), np.nan, dtype=np.float32)
    models = []
    for f in range(cfg.n_folds):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        tr = _subsample(tr, y, cfg.gbdt_max_train_pairs, rng)
        # inner validation split by S1 for early stopping
        inner_s1 = np.unique(s1_idx[tr])
        val_s1 = set(rng.choice(inner_s1, size=max(1, len(inner_s1) // 10), replace=False).tolist())
        is_val = np.fromiter((s in val_s1 for s in s1_idx[tr]), dtype=bool, count=len(tr))
        tr_fit, tr_val = tr[~is_val], tr[is_val]
        dtr = lgb.Dataset(X[tr_fit], y[tr_fit], feature_name=feature_names, free_raw_data=False)
        dval = lgb.Dataset(X[tr_val], y[tr_val], reference=dtr, free_raw_data=False)
        model = lgb.train(params, dtr, num_boost_round=num_rounds, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(cfg.gbdt_early_stopping, verbose=False), lgb.log_evaluation(0)])
        oof[te] = model.predict(X[te], num_iteration=model.best_iteration).astype(np.float32)
        models.append(model)
        log.info("[%s] fold %d: train=%d (pos=%d) test=%d best_iter=%d val_logloss=%.5f", tag, f, len(tr_fit),
                 int(y[tr_fit].sum()), len(te), model.best_iteration, model.best_score["valid_0"]["binary_logloss"])
    return oof, models


def predict_models(models: List[lgb.Booster], X: np.ndarray, chunk: int = 5_000_000) -> np.ndarray:
    out = np.zeros(len(X), dtype=np.float32)
    for m in models:
        for s in range(0, len(X), chunk):
            out[s:s + chunk] += m.predict(X[s:s + chunk], num_iteration=m.best_iteration)
    return out / max(1, len(models))


def save_models(models: List[lgb.Booster], d: Path, tag: str, feature_names: List[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(str(d / f"{tag}_fold{i}.txt"), num_iteration=m.best_iteration)
    (d / f"{tag}_features.json").write_text(json.dumps(feature_names))


def load_models(d: Path, tag: str) -> Tuple[List[lgb.Booster], List[str]]:
    models = []
    i = 0
    while (d / f"{tag}_fold{i}.txt").exists():
        models.append(lgb.Booster(model_file=str(d / f"{tag}_fold{i}.txt")))
        i += 1
    if not models:
        raise FileNotFoundError(f"no models for tag {tag} in {d}; run --mode train first")
    names = json.loads((d / f"{tag}_features.json").read_text())
    return models, names


def feature_importance(models: List[lgb.Booster], names: List[str], top: int = 30) -> List[Tuple[str, float]]:
    imp = np.zeros(len(names))
    for m in models:
        imp += m.feature_importance(importance_type="gain")
    imp /= max(1, len(models))
    order = np.argsort(-imp)
    return [(names[i], float(imp[i])) for i in order[:top]]
