import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "examples" / "prepare_audit_inputs.py"


def _source(path, *, naive=False, duplicate=False):
    suffix = "" if naive else "Z"
    frame = pd.DataFrame({
        "timestamp": [
            f"2025-01-01T00:00:00{suffix}",
            f"2025-01-02T00:00:00{suffix}",
            f"2025-02-01T00:00:00{suffix}",
            f"2025-02-02T00:00:00{suffix}",
        ],
        "transaction_id": ["t1", "t2", "a1", "a1" if duplicate else "a2"],
        "customer_id": ["train-a", "train-b", "A", "A"],
        "is_fraud": [0, 0, 0, 0],
        "amount": [1.0, 100.0, 10.0, 20.0],
        "merchant_category": ["retail"] * 4,
    })
    frame.to_csv(path, index=False)


def _run(source, out, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run([
        sys.executable, str(SCRIPT),
        "--source", str(source),
        "--train-start", "2025-01-01", "--train-end", "2025-01-31",
        "--audit-start", "2025-02-01", "--audit-end", "2025-02-28",
        "--minimum", "1", "--exposure-min", "2", "--exposure-max", "2",
        "--out", str(out), *extra,
    ], text=True, capture_output=True, env=env)


def test_cli_configurable_windows_and_private_export_opt_in(tmp_path):
    source = tmp_path / "source.csv"
    _source(source)
    out = tmp_path / "out"
    completed = _run(source, out)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((out / "audit_structure.json").read_text())
    assert report["selected_train"] == ["2025-01-01", "2025-01-31"]
    assert report["selected_audit"] == ["2025-02-01", "2025-02-28"]
    assert report["eligible_entities"] == 1
    assert not (out / "PRIVATE_legitimate_audit_with_context.csv").exists()


def test_cli_rejects_naive_timestamps(tmp_path):
    source = tmp_path / "source.csv"
    _source(source, naive=True)
    completed = _run(source, tmp_path / "out")
    assert completed.returncode != 0
    assert "timezone-aware" in completed.stderr


def test_cli_rejects_duplicate_audit_event_ids(tmp_path):
    source = tmp_path / "source.csv"
    _source(source, duplicate=True)
    completed = _run(source, tmp_path / "out")
    assert completed.returncode != 0
    assert "duplicate event ID" in completed.stderr
