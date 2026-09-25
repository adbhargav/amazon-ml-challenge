"""Address parsing by comma component.

Addresses in the data are comma separated but the components are frequently
*reordered*, abbreviated or missing ("24800 Euclid Avenue, Euclid, OH" vs
"OH, EUCLID AVE, EUCLID").  We therefore classify every component by lookup
(state lexicon, postcode pattern, city lexicon learned from Source 1) instead
of by position, and only what is left over is treated as street text.

Output per address (all lower-case ascii):

    state      canonical state / region code ("" if unknown)
    city       cleaned city ("" if unknown)
    postcode   digits ("" if none)
    street     street tokens without numbers, abbreviations expanded
    hn_raw     house number exactly as written ("00412")
    hn         house number with leading zeros stripped ("412")
    hn_suffix  "a", "bis", "13" (from 570/13) ...
    unit       apartment / suite / floor token
    nums       all digit runs (zeros stripped), space separated, sorted
    tokens     every normalised token of the address (for a generic overlap)
    empty      True if the address is empty / "null"
    landmark   True if it references a landmark (near / opp / behind ...)
    ncomp      number of comma components
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional

from .lexicons import (
    CITY_AFFIX_RE, LANDMARK_RE, NUMBER_MARKER_RE, ORDINAL_RE, STOP_TOKENS, UNIT_RE,
    _norm_key, postcode_re, state_lex, street_abbr,
)
from .translit import Translit

_NULLISH = {"null", "none", "nan", "n/a", "na", "-", "--", ".", "nil", "unknown", ""}
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s/\-#]")
_HN_RE = re.compile(r"^(?:([a-z]{1,2})[\-/])?(\d+)([a-z])?(?:(?:[/\-])(\d+[a-z]?|[a-z])|\s+(bis|ter|quater))?\b")
_DIGITS_RE = re.compile(r"\d+")
_LEADING_NUM_RE = re.compile(r"^\d")
_STREET_NOISE = {
    "suite", "apt", "apartment", "unit", "floor", "flat", "room", "office", "bldg", "building",
    "near", "opposite", "behind", "beside", "adjacent", "opp", "nr", "next", "to",
    "bis", "ter", "quater",
}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def clean_city(s: str) -> str:
    s = _norm_key(s)
    s = CITY_AFFIX_RE.sub("", s).strip()
    return _WS_RE.sub(" ", s)


def _split_components(raw: str, translit: Translit) -> List[str]:
    raw = unicodedata.normalize("NFKC", raw)
    out = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        p, _, _ = translit.text(p)
        p = strip_accents(p).lower().strip()
        if p in _NULLISH:
            continue
        out.append(p)
    return out


def build_city_lexicon(addresses: Iterable[str], countries: Iterable[str], translit: Translit,
                       min_count: int = 2) -> Dict[str, set]:
    """Learn the city vocabulary per country from (Source 1) addresses.

    The city is taken as the component immediately before the state component
    (or the last non-numeric component when no state is found).
    """
    counts: Dict[str, Counter] = {}
    for raw, country in zip(addresses, countries):
        parts = _split_components(raw, translit)
        if not parts:
            continue
        slex = state_lex(country)
        pre = postcode_re(country)
        city = None
        state_pos = None
        for i, p in enumerate(parts):
            if _norm_key(p) in slex:
                state_pos = i
        cand_range = range(state_pos - 1, -1, -1) if state_pos is not None else range(len(parts) - 1, -1, -1)
        for i in cand_range:
            q = _norm_key(parts[i])
            if not q or q in slex or pre.match(q) or q.isdigit():
                continue
            toks = q.split()
            # "columbus 43004" -> drop the postcode token
            toks = [t for t in toks if not pre.match(t)]
            q = " ".join(toks)
            if q and not _LEADING_NUM_RE.match(q):
                city = clean_city(q)
            break
        if city:
            counts.setdefault(country, Counter())[city] += 1
    lex = {c: {k for k, n in cnt.items() if n >= min_count} for c, cnt in counts.items()}
    return lex


def _expand_tokens(text: str, abbr: Dict[str, str]) -> List[str]:
    text = NUMBER_MARKER_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = text.replace("#", " ")
    toks: List[str] = []
    for t in _WS_RE.split(text):
        t = t.strip("-/")
        if not t:
            continue
        if ORDINAL_RE.match(t):
            t = re.sub(r"[a-z]+$", "", t)
        exp = abbr.get(t)
        if exp is not None:
            if exp == "":
                continue
            toks.extend(exp.split())
        else:
            toks.append(t)
    return toks


def parse_address(raw: str, country: str, translit: Translit, city_lex: Dict[str, set]) -> dict:
    empty_rec = {
        "a_state": "", "a_city": "", "a_postcode": "", "a_street": "", "a_street_key": "",
        "a_hn_raw": "", "a_hn": "", "a_hn_suffix": "", "a_unit": "", "a_nums": "",
        "a_tokens": "", "a_empty": True, "a_landmark": False, "a_ncomp": 0,
    }
    if raw is None or raw.strip().lower() in _NULLISH:
        return empty_rec
    parts = _split_components(raw, translit)
    if not parts:
        return empty_rec

    slex = state_lex(country)
    pre = postcode_re(country)
    abbr = street_abbr(country)
    cities = city_lex.get(country, set())

    state = city = postcode = ""
    street_parts: List[str] = []
    city_pos: Optional[int] = None
    classified: List[str] = []
    for i, p in enumerate(parts):
        q = _norm_key(p)
        if not q:
            continue
        if q in slex:
            state = slex[q]
            classified.append("state")
            continue
        toks = q.split()
        # pull a postcode token out of a component: whole component, last token
        # ("columbus 43004"), or first token when the rest is a city/state
        # ("59000 lille").  A leading number followed by street words is a
        # house number, not a postcode.
        pc_tok = None
        if len(toks) == 1 and pre.match(toks[0]):
            pc_tok = toks[0]
        elif len(toks) > 1 and pre.match(toks[-1]) and not pre.match(toks[0]):
            pc_tok = toks[-1]
        elif len(toks) > 1 and pre.match(toks[0]):
            rest = " ".join(toks[1:])
            if rest in slex or clean_city(rest) in cities:
                pc_tok = toks[0]
        if pc_tok is not None:
            postcode = pc_tok.lstrip("0") or "0"
            toks = [t for t in toks if t != pc_tok]
            q = " ".join(toks)
            if not q:
                classified.append("postcode")
                continue
            if q in slex:
                state = slex[q]
                classified.append("state")
                continue
        cq = clean_city(q)
        if cq and cq in cities and not _LEADING_NUM_RE.match(cq):
            # prefer the *last* city-like component (locality, city, state order)
            if city:
                street_parts.append(city)
            city = cq
            city_pos = i
            classified.append("city")
            continue
        street_parts.append(p)
        classified.append("street")

    # house number: first street component that starts with a number ----------
    hn_raw = hn = hn_suffix = ""
    for sp in street_parts:
        cleaned = NUMBER_MARKER_RE.sub("", sp).strip().lstrip("#- ")
        m = _HN_RE.match(cleaned)
        if m and not ORDINAL_RE.match(cleaned.split()[0] if cleaned.split() else ""):
            prefix, digits, letter, tail, word = m.groups()
            hn_raw = digits + (letter or "")
            hn = digits.lstrip("0") or "0"
            hn_suffix = "".join(x for x in (prefix, letter, tail, word) if x)
            break

    unit = ""
    um = UNIT_RE.search(" ".join(parts))
    if um:
        unit = um.group(1).lower()

    street_tokens: List[str] = []
    for sp in street_parts:
        for t in _expand_tokens(sp, abbr):
            if _DIGITS_RE.search(t) or t in STOP_TOKENS or t in _STREET_NOISE:
                continue
            street_tokens.append(t)
    all_text = " ".join(parts)
    nums = sorted({(n.lstrip("0") or "0") for n in _DIGITS_RE.findall(all_text)})
    landmark = bool(LANDMARK_RE.search(all_text))

    tokens = list(street_tokens)
    if city:
        tokens.extend(city.split())
    if state:
        tokens.append(f"st_{state.lower()}")
    if postcode:
        tokens.append(f"pc_{postcode}")
    tokens.extend(nums)

    return {
        "a_state": state,
        "a_city": city,
        "a_postcode": postcode,
        "a_street": " ".join(street_tokens),
        "a_street_key": " ".join(sorted(set(street_tokens))),
        "a_hn_raw": hn_raw,
        "a_hn": hn,
        "a_hn_suffix": hn_suffix,
        "a_unit": unit,
        "a_nums": " ".join(nums),
        "a_tokens": " ".join(dict.fromkeys(tokens)),
        "a_empty": False,
        "a_landmark": landmark,
        "a_ncomp": len(parts),
    }
