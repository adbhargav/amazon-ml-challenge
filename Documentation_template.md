# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

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
  (`Near SBI ATM`); ~3.5% of S2/S3 addresses are empty; house numbers carry leading
  zeros, truncations and ±1 errors while ~18% of true pairs share no number at all.
* **Name noise**: suffix variants (`Pvt Ltd`/`Private Limited`), dropped suffixes, word
  order shuffles, typos (40% of Latin pairs need 6+ edits), accents, l33t digits
  (`5ky`, `C0rnerstone`), concatenated domain names (`elevateimpex.com`), `dba`/`M/s`
  aliases, junk prefixes, and — for India — names written in Devanagari, Tamil, Telugu,
  Kannada, Bengali, Gujarati, Malayalam or Gurmukhi as word-by-word renderings of the
  English name.
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

---

## 3. Candidate Generation (Blocking)

Everything runs inside a (country, state) partition; pool records whose state could not
be parsed (~5%) are added to every partition of their country, and S1 records without
a parsed state search the whole country.  Five retrieval paths are unioned, each
contributing its score and rank as features:

- **Blocking keys used:**
  1. char-3-gram TF-IDF cosine top-20 on the normalised core name (+ alternative
     renderings: transliteration runner-up, alias halves),
  2. char-3-gram TF-IDF top-20 on `core name | street city`,
  3. char-3-gram TF-IDF top-10 on the address alone (`house-number street city
     postcode`) — this rescues pairs whose names share nothing (cross-script, domain
     names, `dba` names),
  4. exact keys: (sorted core-name key, same-or-missing state), (city, street key),
     (house number, street key), each skipping blocks larger than 300 records,
  5. a reverse pass: for every pool record its top-3 S1 by path 2, which also provides
     the competition set for the one-to-one step.
  Very frequent n-grams (> 2% of records) are dropped from the vectoriser; they carry
  no IDF weight but dominate the cost of the sparse product.
- **Learned pruning:** a small LightGBM over ~40 cheap features (retrieval scores and
  ranks, token / IDF-weighted overlaps of name and address, house-number / city /
  state / postcode equality, name-frequency counts, candidate source and script) keeps
  the top-20 candidates per S1 with p ≥ 0.003.  This pruned set is exactly what the
  matching models score and is what `candidate_pairs.tsv` contains.
- **Candidate pairs generated:** [fill from `work/test/report.json → candidates.n_pairs`]
  (≈ 5 per S1 after pruning on synthetic data; union ≈ 60–70 per S1 before pruning)
- **How you ensured true matches were not lost:** recall of every path and of the union
  is measured on the ground truth of the queried training S1 (`report.json →
  blocking`), and the pruner's K is chosen so that candidate recall stays ≥ 99%:
  [fill: union recall = …, candidate recall @K=20 = …].  Blocking misses are dumped to
  `errors_blocking_miss.tsv` and inspected by noise bucket.

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
country.

**Threshold selection method:** on the out-of-fold predictions of a *test-like*
validation (15% of S1 hidden so that their records act as distractors) we grid-search
(a) one-to-one assignment with margin δ ∈ {0, 0.05, 0.1}, (b) a probability adjustment
p' = σ(a·logit p + b), and (c) the decision rule — per-S1 expected-F0.5 subset
(Monte-Carlo, 256 samples, k = 0 = singleton) versus a global threshold — to maximise
the actual macro F0.5.  The expected-F0.5 rule wins on validation; the chosen
parameters are stored in `work/models/decision.json` and applied unchanged to the
test split.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [fill from `work/train/report.json → decision.validation.all`]
  (validation on the real training data; the synthetic development set scores 0.98)
  — public leaderboard: [fill]
- **Common false positives (wrong merges):** [from `errors_fp.tsv`; expected: same-name
  twins on the same street with a nearby house number, singletons with a look-alike
  whose candidate address is empty]
- **Common false negatives (missed matches):** [from `errors_fn.tsv` /
  `errors_blocking_miss.tsv`; expected: heavy typos combined with an empty address,
  Indic tokens outside the learned dictionary, aliases whose both halves differ]

---

## 6. Conclusion

The pipeline turns the two structural facts of the data — names are ambiguous but
addresses are not, and every S2/S3 record belongs to one S1 — into features and
decision rules, and optimises the exact competition metric on a validation that mimics
the test distribution.  Its components are cheap (CPU only, MIT/BSD licences) and each
stage reports its own recall / quality so that error analysis, not guesswork, drives
the remaining iterations.

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
`ber/metrics.py`.  Tests in `tests/` (unit tests + a synthetic end-to-end run).
See its `README.md` for exact commands, runtime and hardware.

### B. Additional Results

[Blocking recall table per retrieval path, feature-importance tables, validation by
country / singletons vs matched, and the decision-rule grid — all available in
`work/train/report.json` after the training run.]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
