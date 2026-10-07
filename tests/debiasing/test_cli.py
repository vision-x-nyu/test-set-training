"""IBP command line on the offline fixture; argument validation; lightweight import."""

import json
import subprocess
import sys

import pytest

from TsT.debiasing.__main__ import main


@pytest.fixture
def offline_vsi(monkeypatch, vsi_df):
    from TsT.benchmarks.vsi.benchmark import VSIBenchmark

    seen = {}

    def fake_load(self, revision=None):
        seen["revision"] = revision
        return vsi_df.copy()

    monkeypatch.setattr(VSIBenchmark, "load_data", fake_load)
    return seen


def test_per_format_run_writes_ids_and_summary(offline_vsi, vsi_df, tmp_path, capsys):
    out = tmp_path / "pf"
    assert (
        main(["-b", "vsi", "--alloc", "per_format", "--budget", "20", "--batch_size", "5", "-k", "3", "-o", str(out)])
        == 0
    )
    assert "removed 20 of 300" in capsys.readouterr().out
    removed = (out / "removed_ids.txt").read_text().split()
    kept = (out / "kept_ids.txt").read_text().split()
    assert len(removed) == 20 and len(kept) == 280
    assert set(removed) | set(kept) == {str(i) for i in vsi_df["id"]}
    s = json.loads((out / "summary.json").read_text())
    assert s["revision"] == offline_vsi["revision"] == "bc96b17cb6be84878a6c1f2e64c24346356e0d04"
    assert s["row_order_sha1"] and s["alloc"] == "per_format" and s["tie_break"] == "id" and s["score"] == "s"
    assert [g["group"] for g in s["groups"]] == ["mc", "num"]
    its = s["groups"][0]["iterations"]
    assert [len(it["removed_ids"]) for it in its] == [5, 5, 2]
    assert all(set(it) >= {"iteration", "size_before", "max_bias", "mean_bias"} for it in its)
    assert [str(i) for g in s["groups"] for it in g["iterations"] for i in it["removed_ids"]] == removed


def test_per_type_budgets_from_id_list(offline_vsi, vsi_df, tmp_path):
    ids = tmp_path / "ids.txt"
    sizes = vsi_df[vsi_df["question_type"] == "object_size_estimation"]["id"].tolist()[:6]
    count = vsi_df[vsi_df["question_type"] == "object_counting"]["id"].tolist()[:3]
    ids.write_text("".join(f"{i}\n" for i in sizes + count))
    out = tmp_path / "pt"
    assert main(["-b", "vsi", "--alloc", "per_type", "--budgets_from", str(ids), "-k", "3", "-o", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["budgets"] == {"object_counting": 3, "object_size_estimation": 6}
    assert s["budgets_source"] == "ids.txt" and s["n_removed"] == 9


def test_explicit_per_format_budgets(offline_vsi, tmp_path):
    out = tmp_path / "pfx"
    assert main(["-b", "vsi", "--alloc", "per_format", "--budgets", "mc=4,num=2", "-k", "3", "-o", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["budgets"] == {"mc": 4, "num": 2} and s["n_removed"] == 6 and s["budgets_source"] == "--budgets"


@pytest.mark.parametrize(
    "args",
    [
        ["--alloc", "per_format"],
        ["--alloc", "per_type", "--budget", "10"],
        ["--alloc", "per_type", "--frac", "0.5", "--budgets_from", "x.txt"],
        ["--alloc", "per_format", "--budget", "10", "--score", "delta"],
        ["--alloc", "per_format", "--budget", "10", "--tie_break", "random"],
        ["--alloc", "per_format", "--budgets", "mc:4"],
        ["--alloc", "global", "--budgets", "mc=4"],
    ],
)
def test_invalid_arguments_exit_2(args, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["-b", "vsi", "-o", str(tmp_path / "x"), *args])
    assert e.value.code == 2


def test_help_and_import_are_lightweight():
    code = (
        "import sys, TsT.debiasing, TsT.debiasing.__main__ as m; m.create_parser(); "
        "print('heavy:', [k for k in ('pandas', 'sklearn', 'torch', 'vllm') if k in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "heavy: []" in out.stdout
    out = subprocess.run([sys.executable, "-m", "TsT.debiasing", "--help"], capture_output=True, text=True, check=True)
    for flag in (
        "--alloc",
        "--budget",
        "--frac",
        "--budgets_from",
        "--batch_size",
        "--score",
        "--tie_break",
        "--output_dir",
    ):
        assert flag in out.stdout
