#!/usr/bin/env python3
"""Generate a synthetic dataset in the challenge's layout (for development / tests).

    python src/make_synthetic.py --out synthetic_data --n-train 3000 --n-test 2000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.synthetic import write_dataset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write_dataset(a.out, a.n_train, a.n_test, a.seed)
    print(f"wrote synthetic dataset to {a.out}/{{train,test}}")
