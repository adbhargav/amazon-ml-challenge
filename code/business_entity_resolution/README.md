# Business Entity Resolution — pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
challenge data using only the provided files (no external data, APIs or lookups).
Every model is a LightGBM gradient-boosted tree (MIT licence, a few MB of
parameters); no neural network is required.

```
raw TSV ─► normalize ─► block (union of 5 retrieval paths) ─► prune GBDT ─► candidate_pairs.tsv
                                                                    │
                                                        ~70 pairwise features
                                                                    ▼
                                                            stage-1 GBDT (p1)
                                                                    ▼
                                            context features (twins / competition / cluster agreement)
                                                                    ▼
                                             stage-2 GBDT ─► isotonic calibration (p)
                                                                    ▼
                               one-to-one assignment ─► per-S1 expected-F0.5 subset ─► matching_results.tsv
```

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
```

Place the challenge files (unchanged) at

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

relative to the repository root (the directory that also holds `utils/validate_submission.py`).
All commands below are run from that root.

## 2. Reproduce end-to-end

```bash
# 1) fit everything on the training split and print the OOF validation report (~hours, see §5)
python code/business_entity_resolution/src/run_pipeline.py --mode train

# 2) apply the fitted models to the test split; writes output/matching_results.tsv and
#    output/candidate_pairs.tsv and runs utils/validate_submission.py on them
python code/business_entity_resolution/src/run_pipeline.py --mode test

# 3) build <team>_submission.zip
scripts/package_submission.sh <team_name>
```

Stages are cached in `work/<mode>/` (parquet / npy) and models in `work/models/`, so
a run can be resumed or partially repeated:

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode train --stages block:decide
python code/business_entity_resolution/src/run_pipeline.py --mode train --set hidden_frac=0 --set k_name=30
```

Stages: `load, normalize, block, prune, features, stage1, context, decide, write`.
Every `Config` field (see `src/ber/config.py`) can be overridden with `--set key=value`
or a `--config file.json`.

`--mode train` writes `work/train/report.json` with blocking recall per retrieval
path, candidate recall, OOF AUC / log-loss of each model, feature importances, the
decision-rule grid search and the final **macro F0.5 on the (test-like) out-of-fold
validation**, overall and per country / singletons / matched.  It also dumps
`errors_fp.tsv`, `errors_fn.tsv` and `errors_blocking_miss.tsv` for error analysis.

## 3. Validation harness

* every training S1 is assigned a fold; all models are trained 5-fold GroupKFold by S1
  and every S1 receives **out-of-fold** predictions, so the whole training set is the
  validation set (no small holdout);
* `hidden_frac` (default 0.15) of the S1 are never queried but their S2/S3 records stay
  in the pool.  This raises the share of pool records that match no queried S1 from
  ~26% to ~37%, which is what the test set's higher pool/S1 ratio implies.  The decision
  rule is tuned on that harder setting;
* the metric implementation (`src/ber/metrics.py`) reproduces the worked example of the
  problem statement (0.714) and is unit tested.

## 4. Tests

```bash
cd code/business_entity_resolution
python -m pytest -q                # unit tests + a 1-minute synthetic end-to-end run
python -m pytest -q -m "not slow"  # unit tests only
```

`src/make_synthetic.py` generates a dataset in the challenge layout that reproduces
the documented noise catalogue; it is what the tests and the runtime benchmark use.

## 5. Resources and runtime

The pipeline is CPU only.  Blocking runs inside (country, state) partitions and every
pair-level stage streams through chunked parquet stores, so memory is bounded by
`chunk_rows` plus the models' training subsample (`prune_max_train_pairs`,
`gbdt_max_train_pairs`).

Measured on a 4-core / 15 GB container with a synthetic split of 80k train S1
(387k records, 68k queried, 4.8M union pairs) and 40k test S1 (190k records):

| stage | train | test |
| --- | ---: | ---: |
| normalize + encode | 22 s | 9 s |
| block (5 paths, union) | 53 s | 25 s |
| prune (cheap features + 5-fold GBDT) | 412 s | 65 s |
| features (~70 columns on 406k pairs) | 4 s | 2 s |
| stage-1 GBDT | 91 s | 4 s |
| context + stage-2 GBDT + calibration | 88 s | 5 s |
| decide (grid search / apply) | 85 s | 1 s |
| **total** | **12 min 40 s** | **1 min 54 s** |

Validation macro F0.5 on that run: 0.978 (OOF); 0.976 on the hidden synthetic test
truth; candidate recall 99.2% at 6 candidates per S1.

Extrapolated to the real data (2.2M / 1.7M S1, 10M pool records) with 16+ cores and
64 GB RAM: a few hours for `--mode train`, about an hour for `--mode test`.  Knobs
that trade recall for time: `k_name`, `k_name_addr`, `k_addr`, `k_reverse`,
`prune_top_k`, `tfidf_max_df`, `exact_block_cap`, `gbdt_max_train_pairs`, `n_jobs`.
If RAM is short, lower `chunk_rows`, `prune_max_train_pairs` and `gbdt_max_train_pairs`.

## 6. Source layout

| file | role |
| --- | --- |
| `src/run_pipeline.py` | stage runner / CLI |
| `src/ber/config.py` | all parameters |
| `src/ber/io_utils.py` | TSV loading (tab separated, no quoting), ground truth parsing |
| `src/ber/lexicons.py` | legal suffixes, street abbreviations, US / India / France state tables |
| `src/ber/translit.py` | Indic → Latin transliteration learned from the ground truth (+ MIT fallback) |
| `src/ber/normalize_name.py` | name cleaning, alias variants, domain word-break, l33t decoding |
| `src/ber/normalize_address.py` | comma-component address parser (state / city / postcode / street / house number) |
| `src/ber/normalize.py` | multiprocess driver, learned resources (transliteration, city lexicon, segmenter) |
| `src/ber/encode.py` | integer token sets, IDF, categorical codes, name-frequency counts |
| `src/ber/blocking.py` | TF-IDF char-3-gram top-k (name, name+address, address, reverse) + exact keys |
| `src/ber/pair_features.py` | numba set overlaps + rapidfuzz string scorers (~70 features) |
| `src/ber/models.py` | LightGBM GroupKFold OOF training, fold-averaged inference |
| `src/ber/context.py` | second-stage features (within-S1, competition, cluster agreement, twins) |
| `src/ber/decide.py` | one-to-one assignment, expected-F0.5 subset selection, tuning |
| `src/ber/submission.py` | TSV writer + validator call |
| `src/ber/metrics.py` | official macro F0.5 |
| `src/ber/synthetic.py` | synthetic data generator for tests / benchmarks |
| `src/score.py` | score a matching_results.tsv against a ground-truth file |
