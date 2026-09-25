"""TSV loading with the quirks of the challenge files handled.

* tab separated, **no quoting** (addresses contain commas and quotes),
* all columns kept as strings, missing -> "",
* ground-truth ID lists are comma separated.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

log = logging.getLogger("ber")

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


@contextmanager
def timer(msg: str) -> Iterator[None]:
    t = time.time()
    log.info("%s ...", msg)
    yield
    log.info("%s done in %.1fs", msg, time.time() - t)


def read_source_tsv(path: str | Path) -> pl.DataFrame:
    """Read a *_sourceN.tsv file into a DataFrame with the 4 canonical columns."""
    df = pl.read_csv(
        str(path),
        separator="\t",
        quote_char=None,
        has_header=True,
        infer_schema_length=0,       # everything as Utf8
        null_values=None,
        try_parse_dates=False,
        encoding="utf8",
    )
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {df.columns}")
    df = df.select(SOURCE_COLS)
    df = df.with_columns([pl.col(c).fill_null("").str.strip_chars() for c in SOURCE_COLS])
    if df["entity_id"].n_unique() != df.height:
        raise ValueError(f"{path}: entity_id is not unique")
    return df


def read_ground_truth(path: str | Path) -> pl.DataFrame:
    """Return DataFrame(source1_entity_id, matched_entity_ids: list[str]).

    Also reads ``matching_results.tsv`` / ``candidate_pairs.tsv`` (same layout);
    the second column is always returned as ``matched_entity_ids``.
    """
    df = pl.read_csv(
        str(path), separator="\t", quote_char=None, has_header=True,
        infer_schema_length=0, encoding="utf8",
    )
    if "source1_entity_id" not in df.columns or len(df.columns) < 2:
        raise ValueError(f"{path}: expected columns source1_entity_id, matched_entity_ids")
    second = "matched_entity_ids" if "matched_entity_ids" in df.columns else df.columns[1]
    df = df.select(["source1_entity_id", pl.col(second).alias("matched_entity_ids")]).with_columns(
        pl.col("matched_entity_ids").fill_null("").str.strip_chars()
    )
    df = df.with_columns(
        pl.when(pl.col("matched_entity_ids") == "")
        .then(pl.lit([]).cast(pl.List(pl.Utf8)))
        .otherwise(pl.col("matched_entity_ids").str.split(","))
        .alias("matched_entity_ids")
    )
    return df


def load_split(cfg) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Load S1, pool (S2 ∪ S3) and optional ground truth for ``cfg.mode``.

    The pool DataFrame gets a ``source`` column (2 or 3) and both frames get a
    dense integer ``idx`` (row position) that is used everywhere downstream
    instead of the string id.
    """
    with timer(f"loading {cfg.mode} split from {cfg.split_dir}"):
        s1 = read_source_tsv(cfg.source_path(1)).with_columns(pl.lit(1, dtype=pl.Int8).alias("source"))
        s2 = read_source_tsv(cfg.source_path(2)).with_columns(pl.lit(2, dtype=pl.Int8).alias("source"))
        s3 = read_source_tsv(cfg.source_path(3)).with_columns(pl.lit(3, dtype=pl.Int8).alias("source"))
        pool = pl.concat([s2, s3], how="vertical")
        s1 = s1.with_row_index("idx")
        pool = pool.with_row_index("idx")
        gt = None
        if cfg.ground_truth_path.exists():
            gt = read_ground_truth(cfg.ground_truth_path)
            missing = set(s1["entity_id"].to_list()) - set(gt["source1_entity_id"].to_list())
            if missing:
                log.warning("%d S1 entities have no ground-truth row; treated as singletons", len(missing))
        for name, df in (("S1", s1), ("pool", pool)):
            log.info("%s: %d rows, countries=%s", name, df.height,
                     dict(df.group_by("country").len().sort("country").iter_rows()))
        if s1.filter(~pl.col("entity_id").str.starts_with("S1-")).height:
            log.warning("some S1 ids do not start with 'S1-'")
    return s1, pool, gt


def gt_to_pairs(gt: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> np.ndarray:
    """Convert ground truth to an int array of shape (n_pairs, 2) = (s1_idx, pool_idx).

    Pairs whose ids are not found are dropped with a warning.
    """
    s1_map = dict(zip(s1["entity_id"].to_list(), s1["idx"].to_list()))
    pool_map = dict(zip(pool["entity_id"].to_list(), pool["idx"].to_list()))
    a, b, dropped = [], [], 0
    for sid, mids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        si = s1_map.get(sid)
        if si is None:
            dropped += len(mids)
            continue
        for m in mids:
            pi = pool_map.get(m)
            if pi is None:
                dropped += 1
                continue
            a.append(si)
            b.append(pi)
    if dropped:
        log.warning("dropped %d ground-truth pairs with unknown ids", dropped)
    if not a:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)], axis=1)


def save_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path), compression="zstd")


def load_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(str(path))
