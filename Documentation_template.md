# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** vajra  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-26

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

On the real training data the pipeline reaches **macro F0.5 = 0.979** out-of-fold
(precision 0.996, recall 0.948; 0.983 US, 0.977 India) on a validation that keeps the
*true local density* of look-alike businesses (whole states, all their records) and
hides 15% of the Source-1 entities so that their records act as distractors, as in the
test set.  The whole submission was produced on a 6-core / 8 GB laptop in 2 h 45 min.

A first version trained on a random 14% sample of the entities scored 0.988 on its own
validation but only 0.9275 on the public leaderboard: random sampling had thinned the
pool 7× and with it the number of same-street, similar-name neighbours each entity
competes with, so the models merged "twins" that only differ by house number and one
word.  The state-based subsample of §2.3 closes most of that gap: this version scores
**0.9426 on the public leaderboard** (rank 826), i.e. +1.5 points on the same test
data, with 3.7 points of validation-to-leaderboard gap remaining (§5).

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
* **Twins**: the pool contains, per Source-1 entity, on average 0.9 (India) to 1.8 (US)
  records on the *same street* with a name similarity ≥ 80 but a *different* house
  number.  Whether such a record is the same business with a noisy number or a
  different business next door is the central difficulty of the data; the ground
  truth treats most of them as different businesses.
* **Address noise**: comma components are reordered, abbreviated (`Ave`/`Avenue`,
  `R.`/`Rue`), dropped (no PIN, no state), padded with `null`, or replaced by landmarks
  (`Near SBI ATM`); 3.3% of S2/S3 addresses are empty; house numbers carry leading
  zeros, truncations and ±1 errors while ~18% of true pairs share no number at all.
  After parsing, a state is found for 94% of pool records and a city for 84%.
* **Name noise**: suffix variants (`Pvt Ltd`/`Private Limited`), dropped suffixes, word
  order shuffles, typos (40% of Latin pairs need 6+ edits), accents, l33t digits
  (`5ky`, `C0rnerstone`), concatenated domain names (`elevateimpex.com`, 3.6% of the
  pool), `dba`/`M/s` aliases, junk prefixes, and — for India — 7–18% of pool names
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
metric-optimal match list, including the singleton decision; (4) a validation design
that reproduces the real density of competing records (whole states) so that the
decision rule is tuned under test-like competition.

Pipeline: `normalize → block → prune → features → stage-1 GBDT → context → stage-2 GBDT
→ calibration → one-to-one → expected-F0.5 subset → output`.

### 2.3 Compute setup and training subsample

The submission was produced on a 6-core Apple-silicon laptop with 8 GB of RAM, which
dictated the following (all reproducible with the scripts in `scripts/`):

* **Normaliser resources fitted on the full train split**
  (`scripts/fit_normalizer_full.py`, streaming): the transliteration dictionary is
  learned from every ground-truth pair with an Indic-script record (1.36 k word
  entries), the word segmenter from all Latin names, the city lexicon from all
  Source-1 addresses.  `--set normalizer_dir=work/normalizer_full` makes training
  load these instead of refitting on the subsample.
* **Models trained on whole states** (`scripts/make_train_subsample_by_state.py`):
  every Source-1 entity of Delhi, Karnataka, Tamil Nadu and Gujarat (35% of India,
  chosen to cover the Devanagari, Kannada, Tamil and Gujarati scripts) and of five
  hash-chosen US states (CO, ME, MO, TX, WA; 16.5% of the US), with *all* S2/S3
  records of those states — matched to a kept entity, matched to an entity of the
  state that is not queried, or unmatched — plus a proportional sample of the
  records with no parsable state.  530,507 S1 and 2.53 M pool records, pool/S1 ratio
  4.77 vs 4.68 in the full split.  Because blocking runs inside (country, state)
  partitions, every kept entity competes against exactly the records it would compete
  against in the full data, which is what makes the validation faithful (§5).
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
  the top-20 candidates per S1 with p ≥ 0.003 (OOF AUC 0.9992).  This pruned set is
  exactly what the matching models score and is what `candidate_pairs.tsv` contains.
- **Candidate pairs generated (test):** 12,427,209 for 1,732,544 S1 = 7.2 per S1
  (France 8.2, US 7.0, India 7.0) from a union of 61.8 M pairs (France 11.9 M,
  US 21.0 M, India 28.9 M; 32–46 per S1).  On the training validation the union holds
  32.0 pairs per S1 and pruning keeps 6.1.
- **How you ensured true matches were not lost:** recall of every path and of the union
  is measured on the ground truth of the queried training S1 (`work/train/report.json →
  blocking`): union recall **96.8%** (name 67.0%, name+address 85.2%, address 71.1%,
  reverse 88.4%, exact name key 67.2%, city+street 46.2%, house-number+street 40.0%),
  candidate recall after pruning **96.8%** at K = 20 (92.1% at K = 5, 96.7% from K = 10
  on), i.e. the pruner loses nothing.  The 3.2% lost in blocking is the price of the
  reduced top-k in the densest partitions (Delhi, Karnataka, Texas) on the 8 GB
  machine; with the default depth the union recall on the same data is higher and the
  knobs (`k_name`, `k_name_addr`, `k_addr`, `k_reverse`, `max_union_per_s1`) are the
  first thing to raise on a larger machine.  Blocking misses are dumped to
  `errors_blocking_miss.tsv` and inspected by noise bucket (see §5).

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
country.  Stage 1 reaches OOF AUC 0.9985 / log-loss 0.045; stage 2 improves this to
AUC 0.9990 / log-loss 0.035.  The most important stage-1 features are the pruner
probability, IDF-weighted number overlap, the partial name ratio, the suffix relation
and the street token-set ratio; in stage 2 the competition margin (how much better
this candidate fits *another* S1) dominates, followed by the stage-1 logit, the
candidate's probability share within its S1 and the best probability for another S1.

**Threshold selection method:** on the out-of-fold predictions of a *test-like*
validation (whole states, 15% of S1 hidden so that their records act as distractors)
we grid-search (a) one-to-one assignment with margin δ ∈ {0, 0.05, 0.1}, (b) a
probability adjustment p' = σ(a·logit p + b), and (c) the decision rule — per-S1
expected-F0.5 subset (Monte-Carlo, 256 samples, k = 0 = singleton) versus a global
threshold — to maximise the actual macro F0.5.  The winner is the expected-F0.5 rule
with one-to-one assignment, δ = 0.05, a = 1.5, b = −0.5 (macro F0.5 0.9788 vs 0.9782
for a plain 0.5 threshold without one-to-one); the parameters are stored in
`work/models/decision.json` and applied unchanged to every test shard.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation on real training data (450,343 queried S1 of the
  state subsample, OOF):**

  | split | macro F0.5 | singletons | matched | precision | recall |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | all | **0.9792** | 0.9825 | 0.9791 | 0.9959 | 0.9481 |
  | US | 0.9830 | 0.9804 | 0.9832 | 0.9955 | 0.9617 |
  | India | 0.9766 | 0.9840 | 0.9762 | 0.9961 | 0.9386 |

  Predicted singleton rate 5.95% vs 5.55% true.  On the test set the pipeline outputs
  5,476,546 matches (3.16 per S1) and leaves 6.4% of S1 empty (France 6.0%, US 6.2%,
  India 6.6%) — the same profile as in training, including for the unseen country.
  Public leaderboard: **0.9426** (rank 826) for this version; 0.9275 for the first
  version (random-entity subsample, §1).

  | version | training subsample | own validation | public LB | gap |
  | --- | --- | ---: | ---: | ---: |
  | v1 | random 14% of entities | 0.988 | 0.9275 | 6.0 |
  | v3 (this) | whole states, full-split normaliser | 0.979 | 0.9426 | 3.7 |

  The remaining gap is not explained by the validation itself (it already hides 15%
  of the entities, which brings the pool-to-queried-entity ratio to 5.6 against 5.75
  in the test split).  Two things the validation cannot see are the leading
  suspects: **France** (15% of the test split, no ground truth at all — a French
  macro F0.5 of ~0.74 with US/India at their validation level would produce exactly
  0.943) and **states outside the nine training states** (the models are trained
  and validated on the same nine states; Telugu, Bengali, Malayalam and Gurmukhi
  candidates were never seen by the GBDTs, only by the transliteration dictionary).
  A per-country probe (upload the same file with the French matches removed; the
  difference divided by the French share of S1 gives the French score) is the next
  submission.
- **Why the first version over-scored its own validation:** a random 14% sample of
  entities keeps only 14% of the pool, so each kept entity had 5× fewer same-street
  look-alikes than in the full data (0.37 vs 1.84 per S1 in the US).  Its models
  learned that a same-street, similar-name pair is almost always a match and merged
  twins on the test set; 11.8% of its kept US matches had a *different* house number
  against 8.4% among true pairs.  The whole-state subsample restores the real
  density; the validation score drops from 0.988 to 0.979 but now describes the test
  set.  A second intermediate version whose hash-chosen Indian states contained no
  Devanagari names learned a 601-entry transliteration dictionary and left 11.5% of
  Indian test entities empty; fitting the normaliser on the full split (§2.3) fixed
  that (6.6%).
- **Common false positives (wrong merges), from `errors_fp.tsv`:** almost all have the
  *same address* as the S1 and a name that differs by a few characters or one token —
  `EFE Institutions Care` vs `EFYE Institutions Care`, `Adl Industries LLP` vs
  `Adl Infratech LLP`, or an unrelated name at the same house number.  These are twin
  businesses at one address, which the ground truth keeps apart while every noise
  pattern we model (typos, dropped tokens) would merge.  A second group is an
  exact-name candidate with an *empty* address when the true match also has an empty
  address.  Both are inherently ambiguous; the one-to-one step and the competition
  features remove most of them, which is why precision is 0.996.
- **Common false negatives (missed matches), from `errors_fn.tsv` and
  `errors_blocking_miss.tsv`:** the decision rule drops pairs at p ≈ 0.75 when the
  candidate has an empty address *and* a name variation, or a house-number edit
  (`2076` vs `20767`, `1677/F-1` vs `1675/F-1`) — under the precision-weighted metric
  this is the right trade-off.  Blocking misses (3.2%) are dominated by heavy name
  typos combined with an empty address (`APPLIED MEDIA SRECAES`), Indic names whose
  tokens are outside the learned dictionary, and aliases whose both halves differ.
  These are exactly the cases the reduced blocking depth (§2.3) affects.

---

## 6. Conclusion

The pipeline turns the two structural facts of the data — names are ambiguous but
addresses are not, and every S2/S3 record belongs to one S1 — into features and
decision rules, and optimises the exact competition metric on a validation that
reproduces the test distribution, including the density of look-alike neighbours.
Its components are cheap (CPU only, MIT/BSD licences), each stage reports its own
recall / quality so that error analysis drives the remaining iterations, and the whole
thing runs end to end on an 8 GB laptop.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `src/run_pipeline.py --mode train` fits the
normaliser resources (or loads them with `--set normalizer_dir=...`), the pruner, the
two GBDT stages, the calibrator and the decision rule and writes the validation
report; `src/run_pipeline.py --mode test` regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` and runs `utils/validate_submission.py`.  Modules:
`ber/normalize_name.py`, `ber/normalize_address.py`, `ber/translit.py`,
`ber/blocking.py`, `ber/pair_features.py`, `ber/context.py`, `ber/models.py`,
`ber/decide.py`, `ber/metrics.py`.  Tests in `tests/` (21 unit tests + a synthetic
end-to-end run).  See its `README.md` for exact commands.

Reproduction of this submission (repository `scripts/`, from the repository root with
the challenge TSVs in `dataset/{train,test}/`):

```
.venv/bin/python scripts/extract_states.py                       # parse the state of every train record
.venv/bin/python scripts/fit_normalizer_full.py --out work/normalizer_full
.venv/bin/python scripts/make_train_subsample_by_state.py --frac 0.15 \
    --must India:DL,India:KA,India:TN,India:GJ --out dataset_state
scripts/split_test_by_country.sh
TRAIN_DIR=dataset_state NORMALIZER_DIR=work/normalizer_full scripts/run_real.sh vajra
```

(`scripts/extract_states.py` needs a fitted normaliser under `work/models/normalizer`;
run `fit_normalizer_full.py` first and point `--models` at its output, or use the one
from any earlier training run — only the parsed state is used.)

### B. Additional Results

Measured runtime on the 6-core / 8 GB laptop (Python 3.14, LightGBM 4.7, polars 1.44):

| run | records | union pairs | wall time | heaviest stages |
| --- | ---: | ---: | ---: | --- |
| train (state subsample) | 530k S1 + 2.53 M pool | 14.4 M | 44.5 min | stage-1 15.2 min, block 8.4 min, prune 8.1 min |
| test France | 259k S1 + 1.43 M pool | 11.9 M | 16.6 min | block 5.0 min, prune 5.0 min |
| test US | 663k S1 + 3.82 M pool | 21.0 M | 35.0 min | block 8.9 min, prune 9.3 min |
| test India | 810k S1 + 4.72 M pool | 28.9 M | 68.4 min | block 31.8 min, prune 12.5 min |

Peak resident memory of the main process stayed under 2 GB thanks to the chunked
parquet feature store; the India TF-IDF index pushed the machine to 6–10 GB of swap
but completed.  Full blocking-recall tables per retrieval path, feature-importance
lists, validation by country and the 66-point decision-rule grid are in
`work/train/report.json` after the training run.
