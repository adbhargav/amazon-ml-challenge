"""Indic-script -> Latin transliteration.

Source 1 is always Latin script, but a large share of Indian Source 2/3
records are written in Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati,
Malayalam or Gurmukhi.  Those names are *word-by-word phonetic renderings of
the same English words* (e.g. "सिल्वर फाउंडेशन प्राइवेट लिमिटेड" ↔ "Silver
Foundation Private Limited"), so the best transliterator is a dictionary
learned from the ground-truth pairs of the training set:

1. for every GT pair whose candidate contains Indic script, split both names on
   whitespace; if the token counts agree, align position-wise and count
   (indic_token -> latin_token);
2. keep the most frequent Latin rendering of every Indic token (plus a runner
   up when it holds >= 20% of the mass, e.g. "प्रा." -> pvt / private);
3. tokens missing from the dictionary fall back to ``indic_transliteration``
   (MIT) ITRANS output, snapped to the closest dictionary word by consonant
   skeleton when one exists.

The dictionary is learned once (``fit``), saved to json, and re-used in test
mode.  No external data is used.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

log = logging.getLogger("ber")

# Unicode block ranges of the scripts present in the data.
_SCRIPT_RANGES: List[Tuple[int, int, str]] = [
    (0x0900, 0x097F, "DEVANAGARI"),
    (0x0980, 0x09FF, "BENGALI"),
    (0x0A00, 0x0A7F, "GURMUKHI"),
    (0x0A80, 0x0AFF, "GUJARATI"),
    (0x0B00, 0x0B7F, "ORIYA"),
    (0x0B80, 0x0BFF, "TAMIL"),
    (0x0C00, 0x0C7F, "TELUGU"),
    (0x0C80, 0x0CFF, "KANNADA"),
    (0x0D00, 0x0D7F, "MALAYALAM"),
]
_INDIC_RE = re.compile("[ऀ-෿]")
_PUNCT_STRIP_RE = re.compile(r"^[\W_]+|[\W_]+$")
_ASCII_PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~]")   # ascii punctuation only (keeps Indic marks)
_VOWEL_RE = re.compile(r"[aeiou]")

try:  # optional MIT dependency
    from indic_transliteration import sanscript as _sanscript
    _SCHEME = {
        "DEVANAGARI": _sanscript.DEVANAGARI, "BENGALI": _sanscript.BENGALI,
        "GURMUKHI": _sanscript.GURMUKHI, "GUJARATI": _sanscript.GUJARATI,
        "ORIYA": _sanscript.ORIYA, "TAMIL": _sanscript.TAMIL, "TELUGU": _sanscript.TELUGU,
        "KANNADA": _sanscript.KANNADA, "MALAYALAM": _sanscript.MALAYALAM,
    }
    _HAVE_INDIC = True
except Exception:  # pragma: no cover
    _HAVE_INDIC = False
    _SCHEME = {}


def has_indic(text: str) -> bool:
    return bool(_INDIC_RE.search(text))


def token_script(tok: str) -> str | None:
    for ch in tok:
        cp = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return name
    return None


def script_class(text: str) -> int:
    """0 = Latin/other only, 1 = Indic only, 2 = mixed."""
    indic = latin = False
    for ch in text:
        if ch.isalpha():
            cp = ord(ch)
            if 0x0900 <= cp <= 0x0DFF:
                indic = True
            else:
                latin = True
            if indic and latin:
                return 2
    return 1 if indic else 0


def _skeleton(word: str) -> str:
    s = _VOWEL_RE.sub("", word.lower())
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def _clean_indic_token(tok: str) -> str:
    """Strip ascii punctuation around/inside a token but keep the script marks."""
    tok = _ASCII_PUNCT_RE.sub("", tok)
    return tok.strip()


class Translit:
    def __init__(self, table: Dict[str, str] | None = None, alt: Dict[str, str] | None = None):
        self.table: Dict[str, str] = table or {}
        self.alt: Dict[str, str] = alt or {}
        self._skel: Dict[str, str] = {}
        self._fallback_cache: Dict[str, str] = {}
        self._rebuild_skeletons()

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, pairs: Iterable[Tuple[str, str]], min_count: int = 2, runner_up_share: float = 0.2) -> "Translit":
        """``pairs`` yields (latin_text, indic_text) from ground-truth matches."""
        counts: Dict[str, Counter] = defaultdict(Counter)
        n_used = 0
        for latin, indic in pairs:
            if not has_indic(indic):
                continue
            lt = [t for t in (_ASCII_PUNCT_RE.sub("", w).lower() for w in latin.split()) if t]
            it = [t for t in (_clean_indic_token(w) for w in indic.split()) if t]
            if not lt or len(lt) != len(it):
                continue
            n_used += 1
            for a, b in zip(it, lt):
                if has_indic(a) and not has_indic(b):
                    counts[a][b] += 1
        table, alt = {}, {}
        for ind, c in counts.items():
            (best, nb), *rest = c.most_common(2)
            total = sum(c.values())
            if nb < min_count:
                continue
            table[ind] = best
            if rest and rest[0][1] / total >= runner_up_share:
                alt[ind] = rest[0][0]
        log.info("translit: learned %d entries (%d alternates) from %d aligned pairs", len(table), len(alt), n_used)
        return cls(table, alt)

    def _rebuild_skeletons(self) -> None:
        self._skel = {}
        for w in set(self.table.values()):
            self._skel.setdefault(_skeleton(w), w)

    # ------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"table": self.table, "alt": self.alt}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "Translit":
        d = json.loads(Path(path).read_text())
        return cls(d.get("table", {}), d.get("alt", {}))

    # --------------------------------------------------------------- apply
    def _fallback(self, tok: str) -> str:
        cached = self._fallback_cache.get(tok)
        if cached is not None:
            return cached
        out = ""
        script = token_script(tok)
        if _HAVE_INDIC and script in _SCHEME:
            try:
                out = _sanscript.transliterate(tok, _SCHEME[script], _sanscript.ITRANS)
            except Exception:  # pragma: no cover
                out = ""
        out = unicodedata.normalize("NFKD", out)
        out = "".join(c for c in out if not unicodedata.combining(c))
        out = re.sub(r"[^a-z0-9]", "", out.lower())
        if out:
            snapped = self._skel.get(_skeleton(out))
            if snapped:
                out = snapped
        self._fallback_cache[tok] = out
        return out

    def token(self, tok: str) -> Tuple[str, bool]:
        """Return (latin_token, was_oov).  Non-Indic tokens are returned unchanged."""
        if not has_indic(tok):
            return tok, False
        clean = _clean_indic_token(tok)
        hit = self.table.get(clean)
        if hit is not None:
            return hit, False
        # mixed token (latin+indic glued): transliterate only the indic part
        return self._fallback(clean), True

    def text(self, text: str) -> Tuple[str, int, int]:
        """Transliterate whitespace-separated tokens. Returns (text, n_indic, n_oov)."""
        if not has_indic(text):
            return text, 0, 0
        out, n_indic, n_oov = [], 0, 0
        for tok in text.split():
            t, oov = self.token(tok)
            if has_indic(tok):
                n_indic += 1
                n_oov += int(oov)
            if t:
                out.append(t)
        return " ".join(out), n_indic, n_oov

    def text_alt(self, text: str) -> str:
        """Variant using the runner-up renderings (empty if no alternates apply)."""
        if not has_indic(text) or not self.alt:
            return ""
        out, changed = [], False
        for tok in text.split():
            clean = _clean_indic_token(tok) if has_indic(tok) else tok
            a = self.alt.get(clean)
            if a is not None:
                out.append(a)
                changed = True
            else:
                out.append(self.token(tok)[0])
        return " ".join(out) if changed else ""
