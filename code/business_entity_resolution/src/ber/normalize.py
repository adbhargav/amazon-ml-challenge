"""Run the name and address normalisers over whole DataFrames (multiprocess)."""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

from .io_utils import timer
from .normalize_address import build_city_lexicon, parse_address
from .normalize_name import WordSegmenter, normalize_name, strip_accents
from .translit import Translit, has_indic

log = logging.getLogger("ber")

NAME_COLS = ["n_full", "n_core", "n_raw_core", "n_alt1", "n_alt2", "n_key", "n_concat", "n_suffix",
             "n_is_domain", "n_has_alias", "n_script", "n_oov", "n_ntok"]
ADDR_COLS = ["a_state", "a_city", "a_postcode", "a_street", "a_street_key", "a_hn_raw", "a_hn",
             "a_hn_suffix", "a_unit", "a_nums", "a_tokens", "a_empty", "a_landmark", "a_ncomp"]

_G: dict = {}


def _init_worker(translit_d: dict, seg_d: dict, city_lex: Dict[str, list]) -> None:
    _G["translit"] = Translit(translit_d["table"], translit_d["alt"])
    _G["segmenter"] = WordSegmenter.from_dict(seg_d)
    _G["city_lex"] = {k: set(v) for k, v in city_lex.items()}


def _work(chunk: List[Tuple[str, str, str]]) -> List[dict]:
    t, seg, cl = _G["translit"], _G["segmenter"], _G["city_lex"]
    out = []
    for name, addr, country in chunk:
        rec = normalize_name(name, t, seg)
        rec.update(parse_address(addr, country, t, cl))
        out.append(rec)
    return out


class Normalizer:
    """Holds the learned resources (transliteration table, segmenter vocabulary,
    city lexicon) and applies them to DataFrames."""

    def __init__(self, translit: Translit, segmenter: WordSegmenter, city_lex: Dict[str, set]):
        self.translit = translit
        self.segmenter = segmenter
        self.city_lex = city_lex

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, s1: pl.DataFrame, pool: pl.DataFrame, gt_pairs, cfg) -> "Normalizer":
        with timer("fitting transliteration dictionary"):
            translit = Translit()
            if gt_pairs is not None and len(gt_pairs):
                s1_names = s1["business_name"].to_list()
                s1_addr = s1["business_address"].to_list()
                pool_names = pool["business_name"].to_list()
                pool_addr = pool["business_address"].to_list()
                indic_mask = [has_indic(n) or has_indic(a) for n, a in zip(pool_names, pool_addr)]

                def gen():
                    for a, b in gt_pairs:
                        if indic_mask[b]:
                            yield s1_names[a], pool_names[b]
                            # address components: align comma components with equal token counts
                            pa, pb = s1_addr[a].split(","), pool_addr[b].split(",")
                            if len(pa) == len(pb):
                                for x, y in zip(pa, pb):
                                    if has_indic(y):
                                        yield x.strip(), y.strip()
                translit = Translit.fit(gen(), cfg.translit_min_count, cfg.translit_runner_up_share)
        with timer("building word segmenter vocabulary"):
            def latin_texts():
                for n in s1["business_name"].to_list():
                    yield strip_accents(n).lower()
                for n in pool["business_name"].to_list():
                    if not has_indic(n) and "." not in n:
                        yield strip_accents(n).lower()
            segmenter = WordSegmenter.from_texts(latin_texts(), min_count=3)
        with timer("building city lexicon from Source 1"):
            city_lex = build_city_lexicon(s1["business_address"].to_list(), s1["country"].to_list(), translit)
            log.info("city lexicon sizes: %s", {k: len(v) for k, v in city_lex.items()})
        return cls(translit, segmenter, city_lex)

    # ------------------------------------------------------------ persist
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.translit.save(d / "translit.json")
        (d / "segmenter.json").write_text(json.dumps(self.segmenter.to_dict()))
        (d / "city_lex.json").write_text(json.dumps({k: sorted(v) for k, v in self.city_lex.items()}, ensure_ascii=False))

    @classmethod
    def load(cls, d: Path) -> "Normalizer":
        translit = Translit.load(d / "translit.json")
        seg = WordSegmenter.from_dict(json.loads((d / "segmenter.json").read_text()))
        cl = {k: set(v) for k, v in json.loads((d / "city_lex.json").read_text()).items()}
        return cls(translit, seg, cl)

    def extend_city_lexicon(self, s1: pl.DataFrame) -> None:
        """In test mode the test Source-1 addresses are unlabeled text of the
        challenge data itself, so the city vocabulary can be extended with them
        (needed for countries unseen in training)."""
        extra = build_city_lexicon(s1["business_address"].to_list(), s1["country"].to_list(), self.translit)
        for c, cities in extra.items():
            self.city_lex.setdefault(c, set()).update(cities)
        log.info("city lexicon sizes after extension: %s", {k: len(v) for k, v in self.city_lex.items()})

    # -------------------------------------------------------------- apply
    def transform(self, df: pl.DataFrame, n_jobs: int = 1, chunk: int = 20_000) -> pl.DataFrame:
        rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(), df["country"].to_list()))
        chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
        init_args = (
            {"table": self.translit.table, "alt": self.translit.alt},
            self.segmenter.to_dict(),
            {k: sorted(v) for k, v in self.city_lex.items()},
        )
        if n_jobs <= 1 or len(rows) < chunk * 2:
            _init_worker(*init_args)
            results = [_work(c) for c in chunks]
        else:
            ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
            with ctx.Pool(n_jobs, initializer=_init_worker, initargs=init_args) as pool:
                results = pool.map(_work, chunks, chunksize=1)
        recs = [r for res in results for r in res]
        out = pl.DataFrame(recs, schema={
            "n_full": pl.Utf8, "n_core": pl.Utf8, "n_raw_core": pl.Utf8, "n_alt1": pl.Utf8, "n_alt2": pl.Utf8,
            "n_key": pl.Utf8, "n_concat": pl.Utf8, "n_suffix": pl.Utf8, "n_is_domain": pl.Boolean,
            "n_has_alias": pl.Boolean, "n_script": pl.Int8, "n_oov": pl.Int16, "n_ntok": pl.Int16,
            "a_state": pl.Utf8, "a_city": pl.Utf8, "a_postcode": pl.Utf8, "a_street": pl.Utf8,
            "a_street_key": pl.Utf8, "a_hn_raw": pl.Utf8, "a_hn": pl.Utf8, "a_hn_suffix": pl.Utf8,
            "a_unit": pl.Utf8, "a_nums": pl.Utf8, "a_tokens": pl.Utf8, "a_empty": pl.Boolean,
            "a_landmark": pl.Boolean, "a_ncomp": pl.Int16,
        })
        return pl.concat([df, out], how="horizontal")
