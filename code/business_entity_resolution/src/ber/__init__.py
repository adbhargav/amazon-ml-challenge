"""Business Entity Resolution pipeline for the Amazon ML Challenge 2026.

Stages (each cached as parquet under ``work/``):

    load  -> normalize -> block -> prune -> features -> stage1 -> context -> decide -> write

Run everything with ``python src/run_pipeline.py --help``.
"""

__version__ = "1.0.0"
