"""Business-name normalisation.

Order matters: transliteration must run *before* accent stripping because
NFKD + combining-mark removal destroys the Devanagari virama / nukta.

For one raw name we produce (all lower-case, ascii):

    full      all tokens after cleaning (legal suffixes canonicalised)
    core      full minus legal-form tokens and stop words   <- main matching key
    alt1/alt2 alternative renderings (alias halves, runner-up transliteration)
    key       sorted unique core tokens joined by space      <- exact blocking key
    concat    core tokens concatenated (no spaces)          <- domain comparison
    raw_core  core without l33t decoding
    suffix    sorted legal-form tokens joined by space
    is_domain the name was a web domain (segmented into words)
    script    0 latin, 1 indic, 2 mixed
    n_oov     indic tokens that were not in the learned dictionary
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional

from .lexicons import (
    ALIAS_RE, DOMAIN_RE, JUNK_PREFIX_RE, JUNK_SUFFIX_RE, LEET_MAP, LEGAL_TOKENS,
    MS_PREFIX_RE, ORDINAL_RE, STOP_TOKENS, SUFFIX_CANON,
)
from .translit import Translit, script_class

_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^a-z0-9\s]")
_HAS_DIGIT_RE = re.compile(r"\d")
_HAS_ALPHA_RE = re.compile(r"[a-z]")


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


class WordSegmenter:
    """Unigram DP word-break for concatenated domain names.

    Cost of a word = -log(p(word)); unknown single characters are allowed at a
    high cost so that typos inside the domain do not break segmentation of the
    surrounding words.
    """

    def __init__(self, counts: Dict[str, int] | None = None, max_len: int = 20):
        self.max_len = max_len
        self.cost: Dict[str, float] = {}
        self.unk_cost = 12.0
        if counts:
            self.fit(counts)

    def fit(self, counts: Dict[str, int]) -> "WordSegmenter":
        total = float(sum(counts.values())) or 1.0
        n = len(counts) or 1
        self.cost = {w: -math.log(c / total) for w, c in counts.items() if w}
        self.unk_cost = -math.log(0.5 / total) + math.log(n) * 0.25
        return self

    @classmethod
    def from_texts(cls, texts: Iterable[str], min_count: int = 2) -> "WordSegmenter":
        c: Counter = Counter()
        for t in texts:
            for tok in t.split():
                if len(tok) > 1 and tok.isalpha():
                    c[tok] += 1
        return cls({w: n for w, n in c.items() if n >= min_count})

    def segment(self, s: str) -> List[str]:
        s = re.sub(r"[^a-z0-9]", "", s.lower())
        n = len(s)
        if n == 0:
            return []
        if not self.cost:
            return [s]
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            lo = max(0, i - self.max_len)
            for j in range(lo, i):
                w = s[j:i]
                c = self.cost.get(w)
                if c is None:
                    if i - j == 1:
                        c = self.unk_cost
                    elif w.isdigit():
                        c = self.unk_cost * 0.5
                    else:
                        continue
                cand = best[j] + c
                if cand < best[i]:
                    best[i] = cand
                    back[i] = j
        out, i = [], n
        while i > 0:
            j = back[i]
            out.append(s[j:i])
            i = j
        out.reverse()
        # glue consecutive unknown single characters back together
        merged: List[str] = []
        for tok in out:
            if merged and len(tok) == 1 and tok not in self.cost and len(merged[-1]) == 1 and merged[-1] not in self.cost:
                merged[-1] += tok
            elif merged and tok not in self.cost and merged[-1] not in self.cost and len(tok) == 1:
                merged[-1] += tok
            else:
                merged.append(tok)
        return merged

    def to_dict(self) -> dict:
        return {"cost": self.cost, "unk_cost": self.unk_cost, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, d: dict) -> "WordSegmenter":
        seg = cls(max_len=d.get("max_len", 20))
        seg.cost = d["cost"]
        seg.unk_cost = d["unk_cost"]
        return seg


def _collapse_single_letters(tokens: List[str]) -> List[str]:
    """'l l c' -> 'llc', 's a r l' -> 'sarl' (only runs of >= 2 single letters)."""
    out: List[str] = []
    run: List[str] = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) >= 2 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) >= 2 else run[0])
    return out


def leet_decode(tok: str) -> str:
    if _HAS_DIGIT_RE.search(tok) and _HAS_ALPHA_RE.search(tok) and not ORDINAL_RE.match(tok):
        return tok.translate(LEET_MAP)
    return tok


def latin_tokens(s: str) -> List[str]:
    """Lower-cased ascii tokens with punctuation removed and suffixes canonicalised."""
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s = s.replace("'", "").replace("’", "")
    s = _NONWORD_RE.sub(" ", s)
    toks = _collapse_single_letters(s.split())
    return [SUFFIX_CANON.get(t, t) for t in toks if t]


def _split_core(tokens: List[str]) -> tuple[List[str], List[str]]:
    core = [t for t in tokens if t not in LEGAL_TOKENS and t not in STOP_TOKENS]
    suffix = sorted({t for t in tokens if t in LEGAL_TOKENS})
    if not core:  # name made only of legal words: keep everything
        core = [t for t in tokens if t not in STOP_TOKENS] or list(tokens)
    return core, suffix


def _clean_raw(raw: str) -> str:
    s = unicodedata.normalize("NFKC", raw).strip()
    s = JUNK_PREFIX_RE.sub("", s)
    s = JUNK_SUFFIX_RE.sub("", s)
    s = MS_PREFIX_RE.sub("", s)
    return s.strip()


def _domain_base(s_latin_lower: str) -> Optional[str]:
    compact = s_latin_lower.replace(" ", "")
    m = DOMAIN_RE.match(compact)
    if not m:
        return None
    return m.group(1).replace("-", "")


def normalize_name(raw: str, translit: Translit, segmenter: WordSegmenter) -> dict:
    s = _clean_raw(raw)
    script = script_class(s)
    n_oov = 0

    # alias variants -----------------------------------------------------
    parts = [p.strip() for p in ALIAS_RE.split(s) if p and p.strip()]
    has_alias = len(parts) > 1
    primary = " ".join(parts) if has_alias else s        # marker word removed
    variants = [primary] + (parts[:2] if has_alias else [])

    # transliterate (before accent stripping) ----------------------------
    tl_variants, alt_tl = [], ""
    for i, v in enumerate(variants):
        t, _, oov = translit.text(v)
        tl_variants.append(t)
        if i == 0:
            n_oov = oov
            alt_tl = translit.text_alt(v)

    def tokens_of(text: str) -> tuple[List[str], bool, str]:
        latin = strip_accents(text).lower()
        base = _domain_base(latin)
        if base is not None:
            seg_src = base
            if _HAS_DIGIT_RE.search(base) and _HAS_ALPHA_RE.search(base):
                seg_src = base.translate(LEET_MAP)        # "tullah0ma" -> "tullahoma" before word-break
            toks = [SUFFIX_CANON.get(t, t) for t in segmenter.segment(seg_src)]
            return toks, True, base
        return latin_tokens(text), False, ""

    full, is_domain, concat_base = tokens_of(tl_variants[0])
    core_raw, suffix = _split_core(full)
    core = [leet_decode(t) for t in core_raw]
    if not is_domain:
        concat_base = "".join(core)

    alts: List[str] = []
    for v in tl_variants[1:]:
        toks, _, _ = tokens_of(v)
        c, _ = _split_core(toks)
        if c:
            alts.append(" ".join(leet_decode(t) for t in c))
    if alt_tl:
        toks, _, _ = tokens_of(alt_tl)
        c, _ = _split_core(toks)
        if c:
            alts.append(" ".join(c))
    alts = [a for a in alts if a and a != " ".join(core)]
    while len(alts) < 2:
        alts.append("")

    return {
        "n_full": " ".join(full),
        "n_core": " ".join(core),
        "n_raw_core": " ".join(core_raw),
        "n_alt1": alts[0],
        "n_alt2": alts[1],
        "n_key": " ".join(sorted(set(core))),
        "n_concat": concat_base,
        "n_suffix": " ".join(suffix),
        "n_is_domain": is_domain,
        "n_has_alias": has_alias,
        "n_script": script,
        "n_oov": n_oov,
        "n_ntok": len(core),
    }
