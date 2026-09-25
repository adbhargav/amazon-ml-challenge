"""Smoke test: synthetic data -> train pipeline -> test pipeline -> validator PASS -> decent score."""
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ber.io_utils import read_ground_truth  # noqa: E402
from ber.metrics import macro_f05  # noqa: E402
from ber.synthetic import write_dataset  # noqa: E402


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    data = tmp_path / "data"
    write_dataset(data, n_train=800, n_test=400, seed=3)
    common = ["--data-dir", str(data), "--work-dir", str(tmp_path / "work"), "--output-dir", str(tmp_path / "out"),
              "--set", "n_jobs=2", "--set", "gbdt_num_rounds=200", "--set", "expected_f_samples=32", "--set", "n_folds=3"]
    r = subprocess.run([sys.executable, str(SRC / "run_pipeline.py"), "--mode", "train", *common], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    r = subprocess.run([sys.executable, str(SRC / "run_pipeline.py"), "--mode", "test", *common], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "PASS" in (r.stdout + r.stderr)

    pred = read_ground_truth(tmp_path / "out" / "matching_results.tsv")
    truth = read_ground_truth(data / "test_ground_truth_HIDDEN.tsv")
    p = dict(zip(pred["source1_entity_id"].to_list(), pred["matched_entity_ids"].to_list()))
    t = dict(zip(truth["source1_entity_id"].to_list(), truth["matched_entity_ids"].to_list()))
    assert set(p) == set(t)
    score = macro_f05(p, t, list(t))
    assert score > 0.85, score
    # candidate file must contain every final match
    cand = read_ground_truth(tmp_path / "out" / "candidate_pairs.tsv")
    c = dict(zip(cand["source1_entity_id"].to_list(), cand["matched_entity_ids"].to_list()))
    for sid, ms in p.items():
        assert set(ms) <= set(c[sid])
