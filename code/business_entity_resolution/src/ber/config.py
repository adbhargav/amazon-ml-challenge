"""Central configuration.

Everything that influences the result is a field here so that a run is fully
described by one ``Config`` object (which ``run_pipeline.py`` dumps to
``work/<mode>/config.json``).
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    data_dir: str = "dataset"           # contains train/ and test/
    work_dir: str = "work"              # parquet caches, models, reports
    output_dir: str = "output"          # matching_results.tsv / candidate_pairs.tsv
    mode: str = "train"                 # "train" (with OOF validation) or "test"

    # ---- reproducibility / resources --------------------------------------
    seed: int = 2026
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)
    chunk_rows: int = 2_000_000         # pair-chunk size for feature computation

    # ---- validation harness -----------------------------------------------
    # Fraction of train S1 that are "hidden": never queried, but their S2/S3
    # records stay in the pool as distractors (simulates the test set, which
    # has ~5.75 S2+S3 records per S1 vs 4.68 in train).
    hidden_frac: float = 0.15
    n_folds: int = 5

    # ---- blocking ----------------------------------------------------------
    ngram_range: tuple = (3, 3)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.02          # drop char n-grams present in > 2% of records (huge speed-up)
    partition_by_state: bool = True     # block within (country, state); missing-state pool records join every partition
    tfidf_max_features: Optional[int] = 400_000
    k_name: int = 20                    # top-k per S1 from the name index
    k_name_addr: int = 20               # top-k per S1 from the name+address index
    k_addr: int = 10                    # top-k per S1 from the address-only index
    k_reverse: int = 3                  # top-k S1 per pool record (reverse pass)
    tfidf_threshold: float = 0.05
    exact_block_cap: int = 300          # skip exact-key blocks bigger than this
    max_union_per_s1: int = 150

    # ---- prune (cheap model that produces candidate_pairs.tsv) ------------
    prune_top_k: int = 20
    prune_min_p: float = 0.003
    prune_num_rounds: int = 400

    # ---- stage-1 / stage-2 GBDT ---------------------------------------------
    gbdt_params: dict = field(default_factory=lambda: dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=255,
        max_depth=-1,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        num_threads=max(1, (os.cpu_count() or 2) - 1),
    ))
    gbdt_num_rounds: int = 3000
    gbdt_early_stopping: int = 100
    gbdt_max_train_pairs: Optional[int] = 25_000_000   # subsample if larger

    # ---- decision ------------------------------------------------------------
    one_to_one: bool = True
    one_to_one_margin: float = 0.0
    decision_rule: str = "auto"         # "expected_f" | "threshold" | "auto"
    threshold: float = 0.5              # used when decision_rule == "threshold"
    expected_f_samples: int = 256
    max_matches_per_s1: int = 12        # GT never exceeds 11

    # ---- misc --------------------------------------------------------------------
    translit_min_count: int = 2
    translit_runner_up_share: float = 0.2

    # ---- derived helpers -----------------------------------------------------
    @property
    def split_dir(self) -> Path:
        return Path(self.data_dir) / self.mode

    @property
    def work(self) -> Path:
        p = Path(self.work_dir) / self.mode
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        p = Path(self.work_dir) / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def source_path(self, k: int) -> Path:
        return self.split_dir / f"{self.mode}_source{k}.tsv"

    @property
    def ground_truth_path(self) -> Path:
        return self.split_dir / f"{self.mode}_ground_truth.tsv"

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        return json.dumps(d, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        clean = {}
        for k, v in d.items():
            if k not in known:
                raise KeyError(f"unknown config key: {k}")
            if k == "ngram_range":
                v = tuple(v)
            clean[k] = v
        return cls(**clean)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))
