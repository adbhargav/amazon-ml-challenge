"""Chunked on-disk feature store so that pair-level stages never hold all pairs in RAM."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator, List

import numpy as np
import polars as pl


class FeatureStore:
    """A directory of ``part-XXXXX.parquet`` files, all with the same columns, in row order."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    # ----------------------------------------------------------------- write
    def reset(self) -> "FeatureStore":
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        return self

    def write_part(self, i: int, df: pl.DataFrame) -> None:
        df.write_parquet(str(self.dir / f"part-{i:05d}.parquet"), compression="zstd")

    # ------------------------------------------------------------------ read
    def parts(self) -> List[Path]:
        return sorted(self.dir.glob("part-*.parquet"))

    @property
    def columns(self) -> List[str]:
        return pl.read_parquet_schema(str(self.parts()[0])).names() if self.parts() else []

    def part_sizes(self) -> np.ndarray:
        return np.array([pl.scan_parquet(str(p)).select(pl.len()).collect().item() for p in self.parts()], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.part_sizes().sum())

    def iter_parts(self) -> Iterator[tuple[int, pl.DataFrame]]:
        """Yield (row_offset, DataFrame) per part."""
        off = 0
        for p in self.parts():
            df = pl.read_parquet(str(p))
            yield off, df
            off += df.height

    def gather(self, rows: np.ndarray) -> pl.DataFrame:
        """Rows (global row indices, any order) as one DataFrame in the order given."""
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        out = []
        sizes = self.part_sizes()
        offsets = np.r_[0, np.cumsum(sizes)]
        for i, p in enumerate(self.parts()):
            lo, hi = offsets[i], offsets[i + 1]
            a, b = np.searchsorted(sorted_rows, [lo, hi])
            if b > a:
                df = pl.read_parquet(str(p))
                out.append(df[sorted_rows[a:b] - lo])
        if not out:
            return pl.DataFrame()
        df = pl.concat(out, how="vertical")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return df[inv]

    def read_all(self) -> pl.DataFrame:
        return pl.concat([df for _, df in self.iter_parts()], how="vertical")
