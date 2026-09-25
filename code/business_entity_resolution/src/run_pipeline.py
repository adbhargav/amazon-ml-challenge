#!/usr/bin/env python3
"""End-to-end pipeline runner.

    python src/run_pipeline.py --mode train            # fit everything + OOF validation report
    python src/run_pipeline.py --mode test             # apply to the test split, write output/*.tsv
    python src/run_pipeline.py --mode train --stages block,prune   # re-run selected stages
    python src/run_pipeline.py --mode train --set hidden_frac=0.0 --set k_name=30

Every stage caches its result under ``work/<mode>/`` and later stages read the
cache, so a run can be resumed or partially repeated.  ``--mode test`` needs
the models produced by ``--mode train`` (``work/models/``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber.blocking import block_all, blocking_recall  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.context import context_features  # noqa: E402
from ber.decide import apply_rule, load_params, save_params, tune  # noqa: E402
from ber.encode import encode, load_enc, save_enc  # noqa: E402
from ber.io_utils import gt_to_pairs, load_parquet, load_split, save_parquet, timer  # noqa: E402
from ber.metrics import macro_f05_from_pairs  # noqa: E402
from ber.models import feature_importance, load_models, predict_models, predict_oof, save_models, subsample_rows, train_folds  # noqa: E402
from ber.store import FeatureStore  # noqa: E402
from ber.normalize import Normalizer  # noqa: E402
from ber.pair_features import cheap_features, full_features  # noqa: E402
from ber.splits import assign_splits  # noqa: E402
from ber.submission import id_lists, run_validator, write_tsv  # noqa: E402

log = logging.getLogger("ber")
STAGES = ["load", "normalize", "block", "prune", "features", "stage1", "context", "decide", "write"]


# ----------------------------------------------------------------- helpers
def pair_labels(pairs: pl.DataFrame, gt_pairs: np.ndarray) -> np.ndarray:
    if gt_pairs is None or len(gt_pairs) == 0:
        return np.zeros(pairs.height, dtype=np.int8)
    key = pairs["s1"].to_numpy().astype(np.int64) * (1 << 32) + pairs["cand"].to_numpy().astype(np.int64)
    gk = gt_pairs[:, 0].astype(np.int64) * (1 << 32) + gt_pairs[:, 1].astype(np.int64)
    return np.isin(key, gk).astype(np.int8)


def _load_gt(cfg: Config):
    p = cfg.work / "gt_pairs.npy"
    return np.load(p) if p.exists() else None


def _queried_mask(cfg: Config, n_s1: int) -> np.ndarray:
    sp = cfg.work / "splits.parquet"
    if not sp.exists():
        return np.ones(n_s1, dtype=bool)
    return ~load_parquet(sp)["hidden"].to_numpy()


def _fold_of_s1(cfg: Config, n_s1: int) -> np.ndarray:
    sp = cfg.work / "splits.parquet"
    if not sp.exists():
        return np.zeros(n_s1, dtype=np.int8)
    return load_parquet(sp)["fold"].to_numpy()


def _report(cfg: Config, key: str, value) -> None:
    p = cfg.work / "report.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = value
    p.write_text(json.dumps(d, indent=2, default=float))
    log.info("report[%s] = %s", key, json.dumps(value, default=float)[:2000])


def _matrix(df: pl.DataFrame) -> np.ndarray:
    return df.to_numpy().astype(np.float32, copy=False)


# ------------------------------------------------------------------ stages
def stage_load(cfg: Config) -> None:
    s1, pool, gt = load_split(cfg)
    save_parquet(s1, cfg.work / "s1.parquet")
    save_parquet(pool, cfg.work / "pool.parquet")
    if gt is not None:
        gt_pairs = gt_to_pairs(gt, s1, pool)
        np.save(cfg.work / "gt_pairs.npy", gt_pairs)
        n_matched = len(np.unique(gt_pairs[:, 0]))
        _report(cfg, "ground_truth", {"n_pairs": int(len(gt_pairs)), "n_s1": s1.height,
                                      "singleton_rate": 1 - n_matched / max(1, s1.height)})
    else:
        p = cfg.work / "gt_pairs.npy"
        if p.exists():
            p.unlink()
    if cfg.mode == "train":
        splits = assign_splits(s1.height, cfg)
        save_parquet(splits, cfg.work / "splits.parquet")
        log.info("splits: hidden=%d / %d S1, %d folds", int(splits["hidden"].sum()), s1.height, cfg.n_folds)
    else:
        p = cfg.work / "splits.parquet"
        if p.exists():
            p.unlink()


def stage_normalize(cfg: Config) -> None:
    s1 = load_parquet(cfg.work / "s1.parquet")
    pool = load_parquet(cfg.work / "pool.parquet")
    nd = cfg.models_dir / "normalizer"
    if cfg.mode == "train":
        if cfg.normalizer_dir:
            log.info("loading pre-fitted normaliser from %s", cfg.normalizer_dir)
            norm = Normalizer.load(Path(cfg.normalizer_dir))
        else:
            norm = Normalizer.fit(s1, pool, _load_gt(cfg), cfg)
        norm.save(nd)
    else:
        norm = Normalizer.load(nd)
        norm.extend_city_lexicon(s1)
    with timer(f"normalising {s1.height + pool.height} records with {cfg.n_jobs} workers"):
        s1n = norm.transform(s1, cfg.n_jobs)
        pooln = norm.transform(pool, cfg.n_jobs)
    save_parquet(s1n, cfg.work / "s1_norm.parquet")
    save_parquet(pooln, cfg.work / "pool_norm.parquet")
    with timer("encoding"):
        e1, e2, idfs = encode(s1n, pooln)
        save_enc(cfg.work / "enc.npz", e1, e2, idfs)
    stats = {
        "pool_indic_share": float((pooln["n_script"] > 0).mean()),
        "pool_oov_tokens": int(pooln["n_oov"].sum()),
        "pool_domain_share": float(pooln["n_is_domain"].mean()),
        "pool_empty_addr_share": float(pooln["a_empty"].mean()),
        "s1_city_found": float((s1n["a_city"] != "").mean()),
        "pool_city_found": float((pooln["a_city"] != "").mean()),
        "s1_state_found": float((s1n["a_state"] != "").mean()),
        "pool_state_found": float((pooln["a_state"] != "").mean()),
        "s1_hn_found": float((s1n["a_hn"] != "").mean()),
        "pool_hn_found": float((pooln["a_hn"] != "").mean()),
    }
    _report(cfg, "normalize", stats)


def stage_block(cfg: Config) -> None:
    s1n = load_parquet(cfg.work / "s1_norm.parquet")
    pooln = load_parquet(cfg.work / "pool_norm.parquet")
    queried = _queried_mask(cfg, s1n.height)
    s1q = s1n.filter(pl.Series(queried))
    with timer(f"blocking {s1q.height} S1 x {pooln.height} pool"):
        pairs = block_all(s1q, pooln, cfg)
    save_parquet(pairs, cfg.work / "pairs_union.parquet")
    rec = blocking_recall(pairs, _load_gt(cfg), queried)
    # per-blocker recall
    gt = _load_gt(cfg)
    if gt is not None and len(gt):
        y = pair_labels(pairs, gt)
        n_gt = int(queried[gt[:, 0]].sum())
        per = {}
        for c in ["sc_name", "sc_na", "sc_addr", "sc_rev"]:
            per[c] = float(((pairs[c].to_numpy() > 0) & (y == 1)).sum() / max(1, n_gt))
        for c in ["ex_name", "ex_street", "ex_hn"]:
            per[c] = float(((pairs[c].to_numpy() > 0) & (y == 1)).sum() / max(1, n_gt))
        rec["recall_by_blocker"] = per
    _report(cfg, "blocking", rec)


def _row_chunks(n: int, size: int):
    for a in range(0, n, size):
        yield a, min(n, a + size)


def _fit_and_score(cfg: Config, store: FeatureStore, y: np.ndarray, fold: np.ndarray, s1: np.ndarray, tag: str,
                   params: dict, rounds: int, max_train: int | None, extra: np.ndarray | None = None,
                   extra_names: list | None = None) -> tuple[np.ndarray, list, list]:
    """Train fold models on a subsample (train mode) or load them (test mode), then score
    every row chunk by chunk.  ``extra`` = additional dense columns aligned with the store rows."""
    names = list(store.columns) + (list(extra_names) if extra_names else [])
    n = len(y)

    def block(df: pl.DataFrame, rows_or_slice) -> np.ndarray:
        X = _matrix(df)
        if extra is not None:
            X = np.concatenate([X, extra[rows_or_slice].astype(np.float32)], axis=1)
        return X

    if cfg.mode == "train":
        rows = subsample_rows(y, max_train, cfg.seed)
        with timer(f"[{tag}] gathering {len(rows)} training rows"):
            X_sub = block(store.gather(rows), rows)
        with timer(f"[{tag}] training {cfg.n_folds} fold models"):
            models = train_folds(X_sub, y[rows], fold[rows], s1[rows], cfg, params, rounds, tag, names)
        del X_sub
        save_models(models, cfg.models_dir, tag, names)
    else:
        models, names_saved = load_models(cfg.models_dir, tag)
        assert list(names_saved) == names, f"{tag}: feature mismatch between train and test"
    p = np.empty(n, dtype=np.float32)
    with timer(f"[{tag}] scoring {n} pairs"):
        for off, df in store.iter_parts():
            sl = slice(off, off + df.height)
            X = block(df, sl)
            p[sl] = predict_oof(models, X, fold[sl]) if cfg.mode == "train" else predict_models(models, X)
    return p, models, names


def stage_prune(cfg: Config) -> None:
    pairs = load_parquet(cfg.work / "pairs_union.parquet")
    e1, e2, idfs = load_enc(cfg.work / "enc.npz")
    s1 = pairs["s1"].to_numpy().astype(np.int64)
    counts = np.bincount(s1, minlength=e1.n)
    store = FeatureStore(cfg.work / "prune_feats").reset()
    with timer(f"cheap features on {pairs.height} union pairs"):
        for i, (a, b) in enumerate(_row_chunks(pairs.height, cfg.chunk_rows)):
            store.write_part(i, cheap_features(pairs[a:b], e1, e2, idfs, counts))
    gt = _load_gt(cfg)
    y = pair_labels(pairs, gt) if cfg.mode == "train" else np.zeros(pairs.height, dtype=np.int8)
    fold = _fold_of_s1(cfg, e1.n)[s1]
    params = dict(cfg.gbdt_params, num_leaves=63, min_data_in_leaf=100, learning_rate=0.1)
    p, models, names = _fit_and_score(cfg, store, y.astype(np.float32), fold, s1, "prune", params,
                                      cfg.prune_num_rounds, cfg.prune_max_train_pairs)
    if cfg.mode == "train":
        _report(cfg, "prune_model", {"oof_auc": float(roc_auc_score(y, p)), "oof_ap": float(average_precision_score(y, p)),
                                     "importance": feature_importance(models, names, 15)})
    pairs = pairs.with_columns([pl.Series("prune_p", p.astype(np.float32)), pl.Series("y", y.astype(np.int8))])
    pairs = pairs.sort(["s1", "prune_p"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("s1").cast(pl.Int16).alias("prune_rank"))
    cands = pairs.filter((pl.col("prune_rank") < cfg.prune_top_k) & (pl.col("prune_p") >= cfg.prune_min_p))
    save_parquet(cands, cfg.work / "candidates.parquet")
    queried = _queried_mask(cfg, e1.n)
    rec = blocking_recall(cands, gt, queried)
    rec["n_pairs_before"] = pairs.height
    if cfg.mode == "train" and gt is not None:
        # recall at several K to guide prune_top_k
        rec["recall_at_k"] = {}
        for k in (5, 10, 15, 20, 30):
            sub = pairs.filter((pl.col("prune_rank") < k) & (pl.col("prune_p") >= cfg.prune_min_p))
            rec["recall_at_k"][str(k)] = blocking_recall(sub, gt, queried)["blocking_recall"]
    _report(cfg, "candidates", rec)


def stage_features(cfg: Config) -> None:
    cands = load_parquet(cfg.work / "candidates.parquet")
    s1n = load_parquet(cfg.work / "s1_norm.parquet")
    pooln = load_parquet(cfg.work / "pool_norm.parquet")
    e1, e2, idfs = load_enc(cfg.work / "enc.npz")
    counts = np.bincount(cands["s1"].to_numpy().astype(np.int64), minlength=e1.n)
    store = FeatureStore(cfg.work / "features").reset()
    with timer(f"full features on {cands.height} candidate pairs"):
        for i, (a, b) in enumerate(_row_chunks(cands.height, cfg.chunk_rows)):
            store.write_part(i, full_features(cands[a:b], s1n, pooln, e1, e2, idfs, cfg, counts))
            log.info("  features chunk %d: rows %d-%d", i, a, b)


def _cand_arrays(cfg: Config):
    cands = load_parquet(cfg.work / "candidates.parquet")
    n_s1 = int(load_parquet(cfg.work / "s1.parquet").height)
    s1 = cands["s1"].to_numpy().astype(np.int64)
    y = cands["y"].to_numpy().astype(np.float32)
    fold = _fold_of_s1(cfg, n_s1)[s1]
    return cands, s1, y, fold


def stage_stage1(cfg: Config) -> None:
    cands, s1, y, fold = _cand_arrays(cfg)
    store = FeatureStore(cfg.work / "features")
    p1, models, names = _fit_and_score(cfg, store, y, fold, s1, "stage1", cfg.gbdt_params, cfg.gbdt_num_rounds, cfg.gbdt_max_train_pairs)
    if cfg.mode == "train":
        _report(cfg, "stage1", {"oof_auc": float(roc_auc_score(y, p1)), "oof_ap": float(average_precision_score(y, p1)),
                                "oof_logloss": float(log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6))),
                                "importance": feature_importance(models, names, 30)})
    np.save(cfg.work / "p1.npy", p1.astype(np.float32))


def stage_context(cfg: Config) -> None:
    cands, s1, y, fold = _cand_arrays(cfg)
    p1 = np.load(cfg.work / "p1.npy")
    s1n = load_parquet(cfg.work / "s1_norm.parquet")
    pooln = load_parquet(cfg.work / "pool_norm.parquet")
    e1, e2, idfs = load_enc(cfg.work / "enc.npz")
    with timer("context features"):
        C = context_features(cands, p1, s1n, pooln, e1, e2, idfs, cfg)
    save_parquet(C, cfg.work / "context.parquet")
    C_names = list(C.columns)
    C_arr = _matrix(C)
    del C
    store = FeatureStore(cfg.work / "features")
    p2, models, names = _fit_and_score(cfg, store, y, fold, s1, "stage2", cfg.gbdt_params, cfg.gbdt_num_rounds,
                                       cfg.gbdt_max_train_pairs, extra=C_arr, extra_names=C_names)
    if cfg.mode == "train":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p2, y)
        joblib.dump(iso, cfg.models_dir / "isotonic.joblib")
        p_cal = iso.predict(p2).astype(np.float32)
        _report(cfg, "stage2", {"oof_auc": float(roc_auc_score(y, p2)), "oof_ap": float(average_precision_score(y, p2)),
                                "oof_logloss": float(log_loss(y, np.clip(p2, 1e-6, 1 - 1e-6))),
                                "oof_logloss_calibrated": float(log_loss(y, np.clip(p_cal, 1e-6, 1 - 1e-6))),
                                "importance": feature_importance(models, names, 30)})
    else:
        iso = joblib.load(cfg.models_dir / "isotonic.joblib")
        p_cal = iso.predict(p2).astype(np.float32)
    np.save(cfg.work / "p2.npy", p2.astype(np.float32))
    np.save(cfg.work / "p_cal.npy", p_cal)


def _breakdown(cfg: Config, keep: np.ndarray, s1: np.ndarray, cand: np.ndarray, gt: np.ndarray, s1_ids: np.ndarray, countries: np.ndarray) -> dict:
    pred = np.stack([s1[keep], cand[keep]], axis=1)
    out = {"all": macro_f05_from_pairs(pred, gt, s1_ids)}
    for c in np.unique(countries[s1_ids]):
        ids = s1_ids[countries[s1_ids] == c]
        out[str(c)] = macro_f05_from_pairs(pred, gt, ids)
    return out


def stage_decide(cfg: Config) -> None:
    cands = load_parquet(cfg.work / "candidates.parquet")
    p = np.load(cfg.work / "p_cal.npy")
    s1 = cands["s1"].to_numpy().astype(np.int64)
    cand = cands["cand"].to_numpy().astype(np.int64)
    s1_df = load_parquet(cfg.work / "s1.parquet")
    queried = _queried_mask(cfg, s1_df.height)
    s1_ids = np.flatnonzero(queried)
    gt = _load_gt(cfg)
    if cfg.mode == "train":
        gt_q = gt[queried[gt[:, 0]]] if gt is not None else np.zeros((0, 2), dtype=np.int64)
        with timer("tuning decision rule on OOF predictions"):
            params, grid = tune(s1, cand, p, gt_q, s1_ids, cfg)
        save_params(params, cfg.models_dir / "decision.json")
        _report(cfg, "decision_grid", grid)
    else:
        params = load_params(cfg.models_dir / "decision.json")
    keep = apply_rule(s1, cand, p, params, cfg)
    matches = cands.filter(pl.Series(keep)).select(["s1", "cand"]).with_columns(pl.Series("p", p[keep]))
    save_parquet(matches, cfg.work / "matches.parquet")
    summary = {"params": params, "n_matches": int(keep.sum()), "n_s1_queried": int(len(s1_ids)),
               "matches_per_s1": float(keep.sum() / max(1, len(s1_ids))),
               "empty_rate": float(1 - len(np.unique(s1[keep])) / max(1, len(s1_ids)))}
    countries = s1_df["country"].to_numpy()
    if gt is not None and cfg.mode == "train":
        gt_q = gt[queried[gt[:, 0]]]
        summary["validation"] = _breakdown(cfg, keep, s1, cand, gt_q, s1_ids, countries)
        # also the plain global-threshold baseline for reference
        base = apply_rule(s1, cand, p, {"rule": "threshold", "threshold": 0.5, "one_to_one": False, "a": 1.0, "b": 0.0}, cfg)
        summary["baseline_thr0.5_no_o2o"] = macro_f05_from_pairs(np.stack([s1[base], cand[base]], 1), gt_q, s1_ids)["macro_f05"]
        _dump_errors(cfg, cands, keep, p, gt_q)
    else:
        per_country = {}
        for c in np.unique(countries):
            ids = np.flatnonzero((countries == c) & queried)
            m = np.isin(s1[keep], ids)
            per_country[str(c)] = {"n_s1": int(len(ids)), "matches_per_s1": float(m.sum() / max(1, len(ids))),
                                   "empty_rate": float(1 - len(np.unique(s1[keep][m])) / max(1, len(ids)))}
        summary["per_country"] = per_country
    _report(cfg, "decision", summary)


def _dump_errors(cfg: Config, cands: pl.DataFrame, keep: np.ndarray, p: np.ndarray, gt: np.ndarray, n: int = 300) -> None:
    """Write a sample of false positives / false negatives with the raw strings for inspection."""
    s1 = load_parquet(cfg.work / "s1.parquet")
    pool = load_parquet(cfg.work / "pool.parquet")
    y = cands["y"].to_numpy()
    df = cands.select(["s1", "cand"]).with_columns([pl.Series("p", p), pl.Series("y", y), pl.Series("kept", keep)])
    fp = df.filter(pl.col("kept") & (pl.col("y") == 0)).sort("p", descending=True).head(n)
    fn = df.filter(~pl.col("kept") & (pl.col("y") == 1)).sort("p", descending=True).head(n)
    for name, part in (("errors_fp", fp), ("errors_fn", fn)):
        out = (part.join(s1.select([pl.col("idx").cast(pl.Int64).alias("s1"), pl.col("entity_id").alias("s1_id"),
                                    pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")]), on="s1")
               .join(pool.select([pl.col("idx").cast(pl.Int64).alias("cand"), pl.col("entity_id").alias("cand_id"),
                                  pl.col("business_name").alias("cand_name"), pl.col("business_address").alias("cand_addr")]), on="cand")
               .select(["p", "s1_id", "s1_name", "s1_addr", "cand_id", "cand_name", "cand_addr"]))
        out.write_csv(str(cfg.work / f"{name}.tsv"), separator="\t")
    # missed ground-truth pairs that never reached the candidate set
    key = cands["s1"].to_numpy().astype(np.int64) * (1 << 32) + cands["cand"].to_numpy().astype(np.int64)
    gk = gt[:, 0].astype(np.int64) * (1 << 32) + gt[:, 1].astype(np.int64)
    missed = gt[~np.isin(gk, key)][:n]
    if len(missed):
        m = pl.DataFrame({"s1": missed[:, 0], "cand": missed[:, 1]})
        out = (m.join(s1.select([pl.col("idx").cast(pl.Int64).alias("s1"), pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")]), on="s1")
               .join(pool.select([pl.col("idx").cast(pl.Int64).alias("cand"), pl.col("business_name").alias("cand_name"), pl.col("business_address").alias("cand_addr")]), on="cand"))
        out.write_csv(str(cfg.work / "errors_blocking_miss.tsv"), separator="\t")


def stage_write(cfg: Config) -> None:
    s1 = load_parquet(cfg.work / "s1.parquet")
    pool = load_parquet(cfg.work / "pool.parquet")
    cands = load_parquet(cfg.work / "candidates.parquet")
    matches = load_parquet(cfg.work / "matches.parquet")
    s1_ids = s1["entity_id"].to_list()
    pool_ids = pool["entity_id"].to_list()
    out_dir = Path(cfg.output_dir) if cfg.mode == "test" else cfg.work / "output"
    order = np.argsort(-matches["p"].to_numpy(), kind="stable")
    m_map = id_lists(matches["s1"].to_numpy(), matches["cand"].to_numpy(), s1_ids, pool_ids, order)
    c_map = id_lists(cands["s1"].to_numpy(), cands["cand"].to_numpy(), s1_ids, pool_ids)
    # every final match must be a candidate (sanity)
    for sid, ms in m_map.items():
        missing = set(ms) - set(c_map.get(sid, []))
        assert not missing, f"match not in candidate set for {sid}: {missing}"
    write_tsv(out_dir / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"], m_map, s1_ids)
    write_tsv(out_dir / "candidate_pairs.tsv", ["source1_entity_id", "candidate_entity_ids"], c_map, s1_ids)
    ok = True
    if cfg.mode == "test":   # the official validator expects the test_* file names
        validator = Path(__file__).resolve().parents[3] / "utils" / "validate_submission.py"
        ok = run_validator(validator, out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv", cfg.split_dir, check_ids=True)
    _report(cfg, "write", {"output_dir": str(out_dir), "validator_pass": bool(ok)})
    if not ok:
        raise SystemExit("submission validation failed")


STAGE_FN = {
    "load": stage_load, "normalize": stage_normalize, "block": stage_block, "prune": stage_prune,
    "features": stage_features, "stage1": stage_stage1, "context": stage_context, "decide": stage_decide, "write": stage_write,
}


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--config", help="json file with Config fields")
    ap.add_argument("--set", action="append", default=[], help="override: key=value (json-parsed)")
    ap.add_argument("--stages", default="all", help="comma list of stages or 'from:to' (e.g. block:decide)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config.from_file(a.config) if a.config else Config()
    cfg.mode = a.mode
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown config key {k}")
        setattr(cfg, k, v)
    if a.data_dir:
        cfg.data_dir = a.data_dir
    if a.work_dir:
        cfg.work_dir = a.work_dir
    if a.output_dir:
        cfg.output_dir = a.output_dir
    if a.stages == "all":
        stages = STAGES
    elif ":" in a.stages:
        lo, hi = a.stages.split(":")
        stages = STAGES[STAGES.index(lo or STAGES[0]): STAGES.index(hi or STAGES[-1]) + 1]
    else:
        stages = [s.strip() for s in a.stages.split(",")]
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return cfg, stages


def main(argv=None) -> None:
    cfg, stages = parse_args(argv)
    (cfg.work / "config.json").write_text(cfg.to_json())
    log.info("mode=%s stages=%s data=%s work=%s", cfg.mode, stages, cfg.split_dir, cfg.work)
    t0 = time.time()
    for st in stages:
        with timer(f"=== stage {st}"):
            STAGE_FN[st](cfg)
    log.info("all done in %.1f min", (time.time() - t0) / 60)
    rp = cfg.work / "report.json"
    if rp.exists():
        d = json.loads(rp.read_text())
        if "decision" in d and "validation" in d["decision"]:
            v = d["decision"]["validation"]["all"]
            log.info("VALIDATION macro F0.5 = %.4f  (singletons %.4f, matched %.4f, precision %.4f, recall %.4f)",
                     v["macro_f05"], v["f05_singletons"], v["f05_matched"], v["micro_precision"], v["micro_recall"])


if __name__ == "__main__":
    main()
