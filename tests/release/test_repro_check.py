"""The no-GPU reproduction of the VSI-Bench-Debiased v1 numbers matches its expected values."""

import subprocess
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).resolve().parents[2] / "reproduce"


def test_reproduce_v1_check():
    script, expected = REPRO / "reproduce_v1.py", REPRO / "expected_v1.json"
    if not (script.exists() and expected.exists()):
        pytest.skip("reproduce/ not present")
    out = subprocess.run([sys.executable, str(script), "--check", str(expected)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
