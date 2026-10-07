"""
Pytest configuration and shared fixtures for TsT tests.

Offline tests use the small stratified text-only subsets in tests/fixtures/
(regenerate with tests/fixtures/make_fixtures.py). Tests marked ``network``
download the full benchmarks from the Hugging Face Hub and are deselected by
default; run them with ``pytest -m network``.
"""

import json
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TsT.core.benchmark import BenchmarkRegistry

FIXTURES = Path(__file__).parent / "fixtures"


def isolate_registry(func):
    """
    Decorator to isolate benchmark registry state during tests.

    Saves the current registry before the test, clears it for clean testing,
    then restores the original state after the test completes (even if it fails).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_benchmarks = BenchmarkRegistry._benchmarks.copy()
        BenchmarkRegistry._benchmarks = {}
        try:
            return func(*args, **kwargs)
        finally:
            BenchmarkRegistry._benchmarks = original_benchmarks

    return wrapper


@pytest.fixture
def clean_registry():
    """Pytest fixture alternative to the decorator (registry cleared, then restored)."""
    original_benchmarks = BenchmarkRegistry._benchmarks.copy()
    BenchmarkRegistry._benchmarks = {}
    try:
        yield
    finally:
        BenchmarkRegistry._benchmarks = original_benchmarks


def load_fixture(name: str, list_col: str) -> pd.DataFrame:
    """Load a JSONL fixture; list columns become object ndarrays, as the HF loader gives them."""
    with open(FIXTURES / name) as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    df[list_col] = df[list_col].apply(lambda x: None if x is None else np.asarray(x, dtype=object))
    return df


@pytest.fixture
def vsi_raw() -> pd.DataFrame:
    """300 VSI-Bench test questions (30 per question type), raw HF columns."""
    return load_fixture("vsi_test_subset.jsonl", "options")


@pytest.fixture
def cvb_raw() -> pd.DataFrame:
    """120 CV-Bench test questions (30 per task), raw HF text columns."""
    return load_fixture("cvb_test_subset.jsonl", "choices")


@pytest.fixture
def vsi_df(vsi_raw) -> pd.DataFrame:
    from TsT.benchmarks.vsi.benchmark import VSIBenchmark

    return VSIBenchmark.preprocess(vsi_raw)


@pytest.fixture
def cvb_df(cvb_raw) -> pd.DataFrame:
    from TsT.benchmarks.cvb.benchmark import CVBBenchmark

    return CVBBenchmark.preprocess(cvb_raw)


@pytest.fixture(autouse=True)
def _reset_tst_logger():
    """The CLI configures the 'TsT' logger for its process; undo that between tests."""
    import logging

    yield
    tst_logger = logging.getLogger("TsT")
    tst_logger.handlers = []
    tst_logger.setLevel(logging.NOTSET)
    tst_logger.propagate = True
