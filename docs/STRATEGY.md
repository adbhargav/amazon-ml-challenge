# Strategy notes

## 1. What the real data looks like (measured by public analyses of the training set)

The numbers below come from participants' EDA of the real files (the data itself is not
in this repository).  The pipeline design is built on them.

| Fact | Consequence in the pipeline |
| --- | --- |
| Train: 2.21M S1, 5.03M S2, 5.29M S3. Test: 1.73M S1, 4.89M S2, 5.08M S3 | Everything is partitioned by (country, state) and streamed through parquet caches; no |S1| x |pool| products |
| 5.6% of S1 are singletons; 0–11 matches per S1 (mode 3–4); ~85% have both an S2 and an S3 match | Per-S1 subset selection instead of a global threshold; cap 12 matches |
| **Every S2/S3 record matches at most one S1** | One-to-one assignment + "competition" features |
| ~26% of pool records match no S1 in train; the test pool/S1 ratio is 5.75 vs 4.68 in train, so the test share is higher | Validation hides 15% of train S1 (their records become distractors); decision parameters tuned there |
| 51% of S1 share a suffix-stripped name key with another S1 elsewhere; ~70% of singletons have a same-name look-alike | Address is the primary discriminator: parsed house number / street / city / state features, name-frequency features |
| Residual hard cases: same street, nearby number (7751 vs 7755); true pairs with number noise (00412, 1515→151, ±1) | Rich house-number features (raw/stripped equality, prefix, |diff|, digit similarity) + cluster-agreement features (do the other S2/S3 candidates agree?) |
| ~18% of true pairs share no number because the candidate dropped it | "missing" is encoded separately from "different" (NaN vs 0) |
| US state agrees in 99.8% of true pairs when both parse; the candidate state is missing in 4.7% | Blocking partitions by state; missing-state pool records join every partition |
| S1 is 100% Latin; 22% of Indian S2 and 12% of Indian S3 names are in Indic scripts, token-aligned renderings of the same English words | Transliteration dictionary learned from the ground truth (covers ~96% of test tokens), `indic_transliteration` (MIT) fallback |
| 5.6% domain names (`pclmedicalcentre.com`), only `.com`, some with l33t digits | Domain detection, l33t decode, DP word-break with a vocabulary learned from the names |
| Alias markers `M/s` (55K), `dba`, `aka`, `t/a`; junk prefixes `# @ *** >> --` | Alias variants, max similarity over variants; junk stripped |
| France = 15% of test, absent from train: SARL/SAS/SASU suffixes, `R.`/`Bd`/`Av` abbreviations, departments (Nord, Gironde) instead of regions | French lexicons; department → region map; `country` is never a feature |
| No ID / row-order leakage | none used |

## 2. Why each design choice

* **Precision-weighted metric with singletons at 1.0** → the decision must be able to
  say "nothing" confidently.  The expected-F0.5 rule scores k = 0 with P(no true match)
  from calibrated probabilities and only adds candidates while the expectation rises.
* **Candidate recall ceiling** → union of five retrieval paths at ≥99% recall, then a
  learned pruner instead of a hand-set cap, so the candidate set handed to the model is
  small (≤20 / S1) but its recall is measured, not assumed.
* **Distribution shift in the test pool** → tune every threshold on the hidden-S1
  validation; compare two leaderboard submissions tuned at `hidden_frac = 0` and `0.15`.
* **Unseen country** → no country feature, country-gated lexicons that are open
  dictionaries, transliteration and city vocabularies that extend to the test S1 text.

## 3. Priority list for the remaining time (highest expected gain first)

1. Run `--mode train` on the real data, read `report.json`: blocking recall per path,
   candidate recall at K = 20, validation F0.5 by country / singletons.  Fix whatever is
   below expectation *before* adding anything (blocking recall < 99% → raise k_* or
   inspect `errors_blocking_miss.tsv`; singleton F0.5 low → the pruner / decision is too
   loose).
2. Submit once; note public LB vs validation.  Submit the `hidden_frac = 0` variant to
   learn which validation setting tracks the LB.
3. Error analysis from `errors_fp.tsv` / `errors_fn.tsv`: bucket by script / domain /
   alias / number mismatch / empty address; add targeted features or lexicon entries.
4. Optional neural boost if a GPU is available (all MIT, < 1B parameters):
   fine-tuned multilingual cross-encoder (`microsoft/mdeberta-v3-base`) scored on the
   pruned candidates and fed as a feature into stage 2; then a bi-encoder
   (`intfloat/multilingual-e5-base`) as an extra blocking path.  Expected +1–3 points on
   the hard tail (cross-script, heavy typos).
5. Seeds / fold ensembling of the GBDTs (cheap, small gain).

## 4. Rules compliance

* Only the provided files are read.  No external APIs, registries, geocoders or
  internet data.  The transliteration dictionary and every vocabulary are learned from
  the challenge data.
* Models: LightGBM (MIT), scikit-learn isotonic regression (BSD).  A few MB of
  parameters — far below the 8B cap.
* `candidate_pairs.tsv` is exactly the set the stage-1/stage-2 models score;
  `matching_results.tsv` is a subset of it (asserted before writing).
