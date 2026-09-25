"""Write ``matching_results.tsv`` and ``candidate_pairs.tsv`` and run the official validator."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

log = logging.getLogger("ber")


def id_lists(pairs_s1: np.ndarray, pairs_cand: np.ndarray, s1_ids: List[str], pool_ids: List[str], order: np.ndarray | None = None) -> Dict[str, List[str]]:
    """Map every S1 id -> list of matched pool ids (ordered by ``order`` if given)."""
    out: Dict[str, List[str]] = {sid: [] for sid in s1_ids}
    s1_ids_arr = np.asarray(s1_ids, dtype=object)
    pool_ids_arr = np.asarray(pool_ids, dtype=object)
    idx = np.arange(len(pairs_s1)) if order is None else order
    seen = set()
    for i in idx:
        key = (int(pairs_s1[i]), int(pairs_cand[i]))
        if key in seen:
            continue
        seen.add(key)
        out[s1_ids_arr[pairs_s1[i]]].append(pool_ids_arr[pairs_cand[i]])
    return out


def write_tsv(path: Path, header: List[str], mapping: Dict[str, List[str]], s1_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{','.join(mapping.get(sid, []))}\n")
    log.info("wrote %s (%d rows, %d non-empty)", path, len(s1_ids), sum(1 for s in s1_ids if mapping.get(s)))


def run_validator(validator: Path, matching: Path, candidate: Path, test_dir: Path, check_ids: bool = False) -> bool:
    if not validator.exists():
        log.warning("validator %s not found; skipping", validator)
        return True
    cmd = [sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(test_dir)]
    if check_ids:
        cmd.append("--check-ids")
    res = subprocess.run(cmd, capture_output=True, text=True)
    log.info("validator output:\n%s", res.stdout.strip())
    if res.returncode != 0:
        log.error("validator FAILED:\n%s", res.stdout + res.stderr)
        return False
    return True
