"""Pinned dataset revisions: every load is checked against the revision's row-order fingerprint.

Offline: the HF loader is replaced by fixtures, so these tests download nothing. The
network counterpart (an offline run after several revisions are cached) is in
test_rf_golden.py.
"""

import importlib
import logging

import pytest

from TsT.__main__ import main
from TsT.benchmarks.cvb.benchmark import CVB_REVISION, CVBBenchmark
from TsT.benchmarks.vsi.benchmark import VSI_REVISION, VSI_REVISION_2025_11_11, VSIBenchmark
from TsT.core.benchmark import DatasetRevisionError
from TsT.utils import row_order_sha1


def test_pinned_fingerprints():
    assert VSIBenchmark.default_revision == VSI_REVISION
    assert VSIBenchmark.row_order_fingerprints[VSI_REVISION] == "8c48d42738671ef11a1ccb6300c81ed5d893f10f"
    assert VSIBenchmark.row_order_fingerprints[VSI_REVISION_2025_11_11] == "b48270fe65bfe142baa12835a230797f6e4f4ec7"
    assert CVBBenchmark.default_revision == CVB_REVISION
    assert CVBBenchmark.row_order_fingerprints[CVB_REVISION] == "02535f1c609db8bba7c6bd64d9a78b9aa3d0fa3a"


def test_resolve_revision():
    bench = VSIBenchmark()
    assert bench.resolve_revision(None) == VSI_REVISION
    assert bench.resolve_revision("bc96b17") == VSI_REVISION  # abbreviation of the pinned sha
    assert bench.resolve_revision(None, mode="llm") == VSI_REVISION_2025_11_11  # TsT-LLM's pin
    assert bench.resolve_revision("d7cb1a3") == VSI_REVISION_2025_11_11  # known revisions expand
    assert bench.resolve_revision("abc1234") == "abc1234"  # anything else is passed through
    assert bench.resolve_revision("main") == "main"


@pytest.fixture
def fake_hub(monkeypatch, vsi_raw, cvb_raw):
    """Replace the HF loader with the fixtures; pin their fingerprints as the default revisions'."""
    served = {"vsi": vsi_raw, "cvb": cvb_raw}
    calls = []

    def fake_load(module_key):
        def load(repo_id, revision, pattern="test*.parquet", drop_columns=()):
            calls.append((repo_id, revision))
            return served[module_key].copy()

        return load

    # the modules (the packages export a `benchmark` instance under the same name)
    vsi_module = importlib.import_module("TsT.benchmarks.vsi.benchmark")
    cvb_module = importlib.import_module("TsT.benchmarks.cvb.benchmark")
    monkeypatch.setattr(vsi_module, "load_hf_split", fake_load("vsi"))
    monkeypatch.setattr(cvb_module, "load_hf_split", fake_load("cvb"))
    monkeypatch.setattr(VSIBenchmark, "row_order_fingerprints", {VSI_REVISION: row_order_sha1(vsi_raw, "id")})
    monkeypatch.setattr(CVBBenchmark, "row_order_fingerprints", {CVB_REVISION: row_order_sha1(cvb_raw, "idx")})
    return served, calls


@pytest.mark.parametrize("bench_cls,key", [(VSIBenchmark, "vsi"), (CVBBenchmark, "cvb")])
def test_matching_rows_load(fake_hub, bench_cls, key):
    served, calls = fake_hub
    df = bench_cls().load_data()
    assert len(df) == len(served[key])
    assert calls[-1][1] == bench_cls.default_revision


@pytest.mark.parametrize("bench_cls,key", [(VSIBenchmark, "vsi"), (CVBBenchmark, "cvb")])
def test_reordered_rows_fail(fake_hub, bench_cls, key):
    served, _ = fake_hub
    served[key] = served[key].iloc[::-1].reset_index(drop=True)  # same questions, other order
    with pytest.raises(DatasetRevisionError, match="do not match revision"):
        bench_cls().load_data()
    with pytest.raises(DatasetRevisionError, match="HF_HUB_OFFLINE"):
        bench_cls().load_data(revision=bench_cls.default_revision[:7])


def test_cli_exits_1_on_mismatch(fake_hub, capsys):
    served, _ = fake_hub
    served["vsi"] = served["vsi"].sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert main(["-b", "vsi", "-k", "3"]) == 1
    captured = capsys.readouterr()
    assert "do not match revision" in captured.err and "weighted mean" not in captured.out


def test_other_revision_logs_its_fingerprint(fake_hub, vsi_raw, caplog):
    with caplog.at_level(logging.WARNING, logger="TsT"):
        VSIBenchmark().load_data(revision="main")
    assert "is not a pinned revision" in caplog.text
    assert row_order_sha1(vsi_raw, "id") in caplog.text
