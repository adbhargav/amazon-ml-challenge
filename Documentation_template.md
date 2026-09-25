# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We resolve Source-1 businesses against Sources 2/3 with a blocking → learned pruning →
two-stage gradient-boosted matcher → per-entity decision pipeline that is built around
the three properties of the data that matter most for macro F0.5: (i) names are heavily
re-used across different businesses so the *address* (parsed into house number, street,
city, state) is the primary discriminator, (ii) every Source-2/3 record belongs to at
most one Source-1 entity, which we exploit with one-to-one assignment and cross-entity
"competition" features, and (iii) the metric rewards a confident "no match", which we
make with a per-entity expected-F0.5 rule on calibrated probabilities.  All text
normalisation (Indic transliteration dictionary, domain-name word-break, alias
variants, address component classification) is learned from the provided data alone.

On the real training data the pipeline reaches **macro F0.5 = 0.988** out-of-fold
(precision 0.998, recall 0.970; 0.991 US, 0.985 India), evaluated on a validation that
hides 15% of the Source-1 entities so that their records act as distractors, as in the
test set.  The whole submission was produced on a 6-core / 8 GB laptop in 2 h 11 min.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings that drove the design (training set):

* **Cardinality**: 5.6% singletons; matched entities have 1–11 links (mode 3–4); 85%
  have links in both S2 and S3.  Roughly a quarter of the S2/S3 pool matches no S1 at
  all, and the test pool is 23% larger relative to S1 than train — so distractors are
  *more* frequent in test.
* **Name ambiguity**: about half of the S1 names (after suffix stripping) are shared by
  another S1 in a different place, and most singletons have a same-name look-alike.
  Name similarity alone therefore produces false merges; the address decides.
* **Address noise**: comma components are reordered, abbreviated (`Ave`/`Avenue`,
  `R.`/`Rue`), dropped (no PIN, no state), padded with `null`, or replaced by landmarks
  (`Near SBI ATM`); 3.3% of S2/S3 addresses are empty; house numbers carry leading
  zeros, truncations and ±1 errors while ~18% of true pairs share no number at all.
  After parsing, a state is found for 94% of pool records and a city for 84%.
* **Name noise**: suffix variants (`Pvt Ltd`/`Private Limited`), dropped suffixes, word
  order shuffles, typos (40% of Latin pairs need 6+ edits), accents, l33t digits
  (`5ky`, `C0rnerstone`), concatenated domain names (`elevateimpex.com`, 3.6% of the
  pool), `dba`/`M/s` aliases, junk prefixes, and — for India — 7.3% of pool names
  written in Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam or
  Gurmukhi as word-by-word renderings of the English name.
* **Country shift**: France (15% of test) never appears in training; it brings its own
  suffixes (SARL, SAS, SASU, SCI), street abbreviations and department/region naming.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruning + two-stage pairwise classifier + per-entity decision rule (Hybrid)  
**Core Innovation:** (1) a transliteration dictionary and address-component parser learned
from the training pairs themselves; (2) a second-stage model over *context* features
(competition between S1s for the same record, agreement between an S1's candidates,
same-name twins) that fixes what pairwise scoring cannot see; (3) a per-S1
expected-F0.5 subset selection that turns calibrated probabilities directly into the
metric-optimal match list, including the singleton decision.

Pipeline: `normalize → block → prune → features → stage-1 GBDT → context → stage-2 GBDT
→ calibration → one-to-one → expected-F0.5 subset → output`.

### 2.3 Compute setup used for this submission

The submission was produced on a 6-core Apple-silicon laptop with 8 GB of RAM, which
dictated three choices (all reproducible with the scripts in `scripts/`):

* **Training on a 14% subsample of the train split** (`scripts/make_train_subsample.py`):
  309,132 Source-1 entities chosen by a deterministic hash of their id, *all* of their
  matched S2/S3 records, and the same 14% share of the S2/S3 records that match no S1
  at all — so the pool/S1 ratio and distractor share of the full split are preserved
  (1.45 M pool records).  Every learned component (transliteration dictionary, city
  lexicon, pruner, both GBDT stages, calibrator, decision rule) is fitted on this
  subsample; the ~1.2 M labelled candidate pairs it yields are far more than the GBDTs
  need.
* **Blocking depth reduced** to top-12 (name), top-12 (name + address), top-6
  (address), top-2 reverse and at most 50 union pairs per S1 (defaults 20/20/10/3/150),
  which keeps the union tables and their sort/group-by inside RAM.  The same settings
  are used in train and test so that rank / count features are distributed identically.
* **Test split processed per country** (`scripts/split_test_by_country.sh`).  Blocking
  never pairs records across countries, so running `--mode test` once per country with
  the shared models and concatenating the outputs is *exactly* equivalent to one run,
  at a third of the peak memory.  Each shard's output was validated with the official
  validator (ID-existence check on), then the concatenation was validated again.

---

## 3. Candidate Generation (Blocking)

Everything runs inside a (country, state) partition; pool records whose state could not
be parsed (~6%) are added to every partition of their country, and S1 records without
a parsed state search the whole country.  Five retrieval paths are unioned, each
contributing its score and rank as features:

- **Blocking keys used:**
  1. char-3-gram TF-IDF cosine top-12 on the normalised core name (+ alternative
     renderings: transliteration runner-up, alias halves),
  2. char-3-gram TF-IDF top-12 on `core name | street city`,
  3. char-3-gram TF-IDF top-6 on the address alone (`house-number street city
     postcode`) — this rescues pairs whose names share nothing (cross-script, domain
     names, `dba` names),
  4. exact keys: (sorted core-name key, same-or-missing state), (city, street key),
     (house number, street key), each skipping blocks larger than 300 records,
  5. a reverse pass: for every pool record its top-2 S1 by path 2, which also provides
     the competition set for the one-to-one step.
  Very frequent n-grams (> 2% of records) are dropped from the vectoriser; they carry
  no IDF weight but dominate the cost of the sparse product.
- **Learned pruning:** a small LightGBM over ~40 cheap features (retrieval scores and
  ranks, token / IDF-weighted overlaps of name and address, house-number / city /
  state / postcode equality, name-frequency counts, candidate source and script) keeps
  the top-20 candidates per S1 with p ≥ 0.003 (OOF AUC 0.9998).  This pruned set is
  exactly what the matching models score and is what `candidate_pairs.tsv` contains.
- **Candidate pairs generated (test):** 12,820,639 for 1,732,544 S1 = 7.4 per S1
  (France 9.8, US 7.0, India 7.0) from a union of 61.8 M pairs (France 11.9 M,
  US 21.0 M, India 28.9 M; 32–46 per S1).  On the training validation the union holds
  32.6 pairs per S1 and pruning keeps 4.6.
- **How you ensured true matches were not lost:** recall of every path and of the union
  is measured on the ground truth of the queried training S1 (`work/train/report.json →
  blocking`): union recall **98.5%** (name 83.6%, name+address 92.2%, address 82.3%,
  reverse 94.9%, exact name key 66.4%, city+street 53.2%, house-number+street 44.5%),
  candidate recall after pruning **98.5%** at K = 20 (93.7% at K = 5, 98.5% from K = 10
  on), i.e. the pruner loses nothing.  The 1.5% lost in blocking is the price of the
  reduced top-k on the 8 GB machine; on the full defaults the synthetic development
  set reaches 99.2%.  Blocking misses are dumped to `errors_blocking_miss.tsv` and
  inspected by noise bucket (see §5).

---

## 4. Matching Model

**Features used (≈70 pairwise + ≈25 context):**
- Name features: token-set / token-sort / partial ratios, Jaro-Winkler and Levenshtein
  on the core name; the same maximised over alias / transliteration variants; Levenshtein
  and partial ratio on the concatenated names (domain names); Jaccard, containment in
  both directions, IDF-weighted Jaccard / containment, maximum IDF of an unmatched S1
  token, sum of IDF of unmatched candidate tokens; suffix relation (same / different /
  one missing / both missing); l33t-decoding gain; exact core / sorted-key equality;
  candidate script, OOV transliteration count, alias and domain flags; name-frequency
  counts (S1 sharing the name key in the country / city, pool records sharing it in
  the city); token counts and ratio.
- Address features: house-number state (both / one / none present), raw and
  zero-stripped equality, prefix/suffix relation, log |difference|, digit similarity,
  suffix (`bis`, `/13`, `a`) and unit equality; number-set Jaccard; street token
  Jaccard / IDF overlap / Jaro-Winkler / token-set ratio; city equality and
  Jaro-Winkler; state and postcode equality (NaN when missing); whole-address token
  Jaccard, IDF overlap and token-set ratio; empty-address and landmark flags;
  component counts.
- Other: the four retrieval scores and ranks, exact-key flags, pruner probability and
  rank, candidate source (S2/S3), number of candidates, interactions (name × address,
  house-number × street, name × house-number).
- Context (stage 2): stage-1 probability and logit; within-S1 rank, gap to the best,
  share of the S1's probability mass, count of candidates with p > 0.5 (overall and per
  source), sum of p (expected cluster size); competition: the candidate's best
  probability for any *other* S1, margin, is-arg-max flag, number of S1s with p > 0.5;
  cluster agreement: address Jaccard, house-number equality and name similarity
  between the candidate and the S1's top-3 other candidates (max and p-weighted mean,
  also restricted to the opposite source); twin flag: a better-scoring candidate with
  the same name key but a different house number, and its gap.

**Model type:** LightGBM binary classifiers (MIT licence; a few MB of parameters), 5-fold
GroupKFold by S1 with out-of-fold predictions for every training pair, fold models
averaged at test time; stage-2 probabilities are isotonic-calibrated on the OOF
predictions.  `country` is never a feature so that France is handled like any other
country.  Stage 1 reaches OOF AUC 0.9986 / log-loss 0.040; stage 2 improves this to
AUC 0.9990 / log-loss 0.031.  The most important stage-1 features are the pruner
probability, the name × address interaction, IDF-weighted number overlap, the suffix
relation and the log house-number difference; in stage 2 the competition margin
(how much better this candidate fits *another* S1) dominates, followed by the stage-1
probability, the gap to the S1's best candidate and the candidate's probability share.

**Threshold selection method:** on the out-of-fold predictions of a *test-like*
validation (15% of S1 hidden so that their records act as distractors) we grid-search
(a) one-to-one assignment with margin δ ∈ {0, 0.05, 0.1}, (b) a probability adjustment
p' = σ(a·logit p + b), and (c) the decision rule — per-S1 expected-F0.5 subset
(Monte-Carlo, 256 samples, k = 0 = singleton) versus a global threshold — to maximise
the actual macro F0.5.  The winner is the expected-F0.5 rule with one-to-one
assignment, δ = 0.1, a = 1.5, b = −0.5 (macro F0.5 0.9883 vs 0.9874 for a plain 0.5
threshold without one-to-one); the parameters are stored in `work/models/decision.json`
and applied unchanged to every test shard.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation on real training data (262,304 queried S1, OOF):**

  | split | macro F0.5 | singletons | matched | precision | recall |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | all | **0.9883** | 0.9922 | 0.9881 | 0.9980 | 0.9702 |
  | US | 0.9908 | 0.9939 | 0.9907 | 0.9982 | 0.9764 |
  | India | 0.9845 | 0.9896 | 0.9842 | 0.9976 | 0.9609 |

  Predicted singleton rate 5.8% vs 5.6% true.  On the test set the pipeline outputs
  5,794,406 matches (3.34 per S1) and leaves 5.3% of S1 empty (France 4.4%, US 5.1%,
  India 5.7%) — the same profile as in training, including for the unseen country.
  Public leaderboard: [fill after submission].
- **Common false positives (wrong merges), from `errors_fp.tsv`:** almost all have the
  *same address* as the S1 and a name that differs by a few characters or one token —
  `EFE Institutions Care` vs `EFYE Institutions Care`, `PM Interstate Bancorporation`
  vs `BUHL Interstate Bancorporation`, `Adl Industries LLP` vs `Adl Infratech LLP`, or
  an unrelated name at the same house number (`St. Cathedral` vs `Umbrafayekelo`,
  317 7th Street).  These are twin businesses at one address, which the ground truth
  keeps apart while every noise pattern we model (typos, dropped tokens) would merge.
  A second group is an exact-name candidate with an *empty* address when the true
  match also has an empty address.  Both are inherently ambiguous; the one-to-one
  step already removes most of them, which is why precision is 0.998.
- **Common false negatives (missed matches), from `errors_fn.tsv` and
  `errors_blocking_miss.tsv`:** the decision rule drops pairs at p ≈ 0.75 when the
  candidate has an empty address *and* a name variation (`Total Gulf` for `Total Gulf
  Bluerock LLC`, `Smt UNIQUE TRUST`), or a house-number edit (`2076` vs `20767`,
  `1677/F-1` vs `1675/F-1`) — the precision-weighted metric makes this the right
  trade-off.  Blocking misses are dominated by heavy name typos combined with an empty
  address (`APPLIED MEDIA SRECAES`, `EAST ADVANCED SACFN LLC`), Indic names whose
  tokens are outside the learned dictionary, and aliases whose both halves differ
  (`Soltavo` for `Vision Group`).  These are exactly the cases the reduced blocking
  depth (§2.3) affects; with more memory `k_name` / `k_name_addr` back at 20 recover
  part of them.

---

## 6. Conclusion

The pipeline turns the two structural facts of the data — names are ambiguous but
addresses are not, and every S2/S3 record belongs to one S1 — into features and
decision rules, and optimises the exact competition metric on a validation that mimics
the test distribution.  Its components are cheap (CPU only, MIT/BSD licences), each
stage reports its own recall / quality so that error analysis drives the remaining
iterations, and the whole thing runs end to end on an 8 GB laptop.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `src/run_pipeline.py --mode train` fits the
normaliser resources, the pruner, the two GBDT stages, the calibrator and the decision
rule and writes the validation report; `src/run_pipeline.py --mode test` regenerates
`output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs
`utils/validate_submission.py`.  Modules: `ber/normalize_name.py`,
`ber/normalize_address.py`, `ber/translit.py`, `ber/blocking.py`,
`ber/pair_features.py`, `ber/context.py`, `ber/models.py`, `ber/decide.py`,
`ber/metrics.py`.  Tests in `tests/` (21 unit tests + a synthetic end-to-end run).
See its `README.md` for exact commands.

Reproduction of this submission (repository `scripts/`):

```
python3 scripts/make_train_subsample.py --rate 0.14 --out dataset_small
scripts/split_test_by_country.sh
scripts/run_real.sh <team_name>        # train, 3 test shards, concatenate, validate, zip
```

### B. Additional Results

Measured runtime on the 6-core / 8 GB laptop (Python 3.14, LightGBM 4.7, polars 1.44):

| run | records | union pairs | wall time | heaviest stages |
| --- | ---: | ---: | ---: | --- |
| train (14% subsample) | 309k S1 + 1.45 M pool | 8.5 M | 20.8 min | prune 7.4 min, stage-1 4.1 min |
| test France | 259k S1 + 1.43 M pool | 11.9 M | 14.0 min | block 5.2 min, prune 4.9 min |
| test US | 663k S1 + 3.82 M pool | 21.0 M | 28.5 min | block 9.1 min, prune 9.1 min |
| test India | 810k S1 + 4.72 M pool | 28.9 M | 66.9 min | block 32.5 min, prune 15.7 min |

Peak resident memory of the main process stayed under 2 GB thanks to the chunked
parquet feature store; the India TF-IDF index pushed the machine to ~7 GB of swap but
completed.  Full blocking-recall tables per retrieval path, feature-importance lists,
validation by country and the 66-point decision-rule grid are in
`work/train/report.json` after the training run.
