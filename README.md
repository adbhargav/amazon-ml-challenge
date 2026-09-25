# Amazon ML Challenge 2026 — Business Entity Resolution

Solution repository for the [Amazon ML Challenge 2026 on Unstop](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604)
(72-hour hackathon, metric: macro F0.5 per Source-1 entity, precision weighted 2x).

The official problem statement is mirrored in [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md),
the official validator in [`utils/validate_submission.py`](utils/validate_submission.py) and the
methodology write-up (submission document) in [`Documentation_template.md`](Documentation_template.md).

```
.
├── code/business_entity_resolution/   # the pipeline (src/, tests/, README.md, requirements.txt)
├── dataset/{train,test}/              # put the challenge TSVs here (git-ignored)
├── output/                            # matching_results.tsv + candidate_pairs.tsv (generated)
├── work/                              # stage caches, models, validation reports (generated)
├── utils/validate_submission.py       # official validator
├── scripts/package_submission.sh      # builds <team>_submission.zip in the required layout
├── docs/                              # problem statement, strategy notes
└── Documentation_template.md          # methodology document for the final package
```

## Quick start (on the machine that has the data)

```bash
git clone <this repo> && cd amazon-ml-challenge
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt

# copy the student_resource dataset into place
cp -r /path/to/student_resource/dataset/train dataset/
cp -r /path/to/student_resource/dataset/test  dataset/

# sanity check the installation (unit tests + 1-minute synthetic end-to-end run)
(cd code/business_entity_resolution && python -m pytest -q)

# 1. train + out-of-fold validation report       -> work/train/report.json
python code/business_entity_resolution/src/run_pipeline.py --mode train
# 2. inference on the test split + validator     -> output/matching_results.tsv, output/candidate_pairs.tsv
python code/business_entity_resolution/src/run_pipeline.py --mode test
# 3. leaderboard: upload output/matching_results.tsv
# 4. final package
scripts/package_submission.sh <team_name>
```

Read `work/train/report.json` after step 1: it holds the blocking recall per retrieval
path, the candidate recall, model AUCs, feature importances and the validation macro
F0.5 (overall, per country, singletons vs matched) that the decision rule was tuned on.
The three `work/train/errors_*.tsv` files list the worst false merges, missed matches
and blocking misses with the raw strings, for the next round of error analysis.

## What the pipeline does (one paragraph)

Names and addresses are normalised with a learned Indic→Latin token dictionary, legal
suffix canonicalisation, alias (dba) variants, l33t decoding and word-break of domain
names, plus a comma-component address parser that classifies each component by lookup
(state / region tables for US, India, France; postcode patterns; a city vocabulary
learned from Source 1) so reordered, abbreviated and partial addresses become
comparable house number / street / city / state fields.  Candidates come from the union
of char-3-gram TF-IDF top-k retrieval on the name, on name+address and on the address
alone, exact keys, and a reverse pass (top S1 per pool record), all run inside
(country, state) partitions; a cheap GBDT prunes the union to ≤20 candidates per S1
(this is `candidate_pairs.tsv`).  A stage-1 LightGBM scores ~70 pairwise features; a
stage-2 LightGBM adds context features — competition between S1s for the same record
(each S2/S3 record matches at most one S1), within-S1 rank/gap, agreement between the
S1's candidates, twin (same name, different house number) flags — and is isotonic
calibrated.  The decision applies one-to-one assignment and then, per S1, the prefix of
candidates that maximises the Monte-Carlo expected F0.5 (k = 0 is the singleton
decision), with the rule's parameters tuned on out-of-fold predictions of a test-like
validation (15% of S1 hidden so their records act as distractors).

## Status

* Built and validated end-to-end on synthetic data that reproduces the documented noise
  catalogue (the real data cannot be downloaded from the environment this was developed
  in).  On an 80k/40k-S1 synthetic split the pipeline scores macro F0.5 = 0.976 on the
  hidden test truth (0.978 OOF validation), the official validator passes, and France
  (absent from training) is matched at the same rate as the training countries.
  Runtime on 4 cores: 12.7 min train, 1.9 min test (table in the code README).
* See `docs/STRATEGY.md` for the measured facts about the real data the design is built
  on, the priority list for the remaining time, and the leaderboard-probing plan.
