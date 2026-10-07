"""Command-line interface: torch-free import and --help, presets, outputs."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from TsT.__main__ import main


def test_help_runs_without_torch():
    code = (
        "import sys, TsT, TsT.__main__ as m; m.create_parser(); "
        "bad = [k for k in ('torch', 'vllm', 'transformers', 'ray') if k in sys.modules]; "
        "print('heavy:', bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "heavy: []" in out.stdout
    out = subprocess.run([sys.executable, "-m", "TsT", "--help"], capture_output=True, text=True, check=True)
    for flag in ("--benchmark", "--revision", "--feature_set", "--group_col", "--output_dir", "--keep_going"):
        assert flag in out.stdout
    assert "Test-set Stress-Test" in out.stdout


def test_import_tst_is_lightweight():
    code = "import sys, TsT; print(TsT.__version__, 'pandas' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["1.0.0", "False"]


def test_cvb_paper_preset_exits_2_before_any_download(capsys):
    assert main(["--benchmark", "cvb", "--feature_set", "paper"]) == 2
    err = capsys.readouterr().err
    assert "75.5" in err and "withdrawn" in err and "README" in err


@pytest.fixture
def offline_vsi(monkeypatch, vsi_df):
    from TsT.benchmarks.vsi.benchmark import VSIBenchmark

    seen = {}

    def fake_load(self, revision=None):
        seen["revision"] = revision
        return vsi_df.copy()

    monkeypatch.setattr(VSIBenchmark, "load_data", fake_load)
    return seen


def test_output_dir_writes_summary_and_s_x(offline_vsi, vsi_df, tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["-b", "vsi", "-k", "3", "--group_col", "scene_name", "-o", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "weighted mean:" in stdout and "bc96b17" in stdout

    summary = json.loads((out / "summary.json").read_text())
    assert summary["revision"] == offline_vsi["revision"] == "bc96b17cb6be84878a6c1f2e64c24346356e0d04"
    assert summary["feature_set"] == "default" and summary["complete"] is True
    assert summary["folds"] == {
        "n_splits": 3,
        "group_col": "scene_name",
        "repeats": 1,
        "random_state": 42,
        "fold_seed": None,
    }
    assert summary["n_rows"] == len(vsi_df) and summary["total_count"] == len(vsi_df)
    assert 0 < summary["weighted_mean"] < 1
    assert len(summary["per_question_type"]) == 8
    by_name = {p["question_type"]: p for p in summary["per_question_type"]}
    assert by_name["obj_appearance_order"]["target_col"] == "gt_idx"
    assert by_name["object_counting"]["target_col"] == "ground_truth"

    s_x = pd.read_csv(out / "s_x.csv")
    assert list(s_x.columns) == [
        "id",
        "model",
        "question_type",
        "format",
        "metric",
        "n_options",
        "chance",
        "s",
        "s_minus_chance",
        "n_repeats",
    ]
    assert s_x["id"].is_unique and set(s_x["id"]) == set(vsi_df["id"])


def test_paper_preset_prints_warning(offline_vsi, capsys):
    assert main(["-b", "vsi", "-k", "3", "--feature_set", "paper", "-q", "object_rel_distance"]) == 0
    err = capsys.readouterr().err
    assert err.count("WARNING: --feature_set paper") == 1


def test_revision_override_is_passed_through(offline_vsi):
    assert main(["-b", "vsi", "-k", "3", "-q", "route_planning", "--revision", "main"]) == 0
    assert offline_vsi["revision"] == "main"


def test_question_types_partial_typo_exits_2(offline_vsi, capsys):
    assert main(["-b", "vsi", "-q", "object_counting,object_rel_direction_eazy"]) == 2
    err = capsys.readouterr().err
    assert "object_rel_direction_eazy" in err and "Valid names" in err and "object_counting" in err
    assert "revision" not in offline_vsi  # rejected before any data is loaded


def test_question_types_unknown_alone_exits_2(offline_vsi, capsys):
    assert main(["-b", "vsi", "-q", "not_a_type"]) == 2
    assert "Unknown question type(s): not_a_type" in capsys.readouterr().err


def test_question_types_accepts_dataset_subtypes(offline_vsi, vsi_df, capsys):
    assert main(["-b", "vsi", "-k", "3", "-q", "object_rel_direction_easy"]) == 0
    captured = capsys.readouterr()
    assert "object_rel_direction" in captured.out and "route_planning" not in captured.out
    assert "scored by the 'object_rel_direction' model" in captured.err
    n_reldir = int(vsi_df["question_type"].str.startswith("object_rel_direction").sum())
    assert f"(questions scored: {n_reldir})" in captured.out


def test_run_evaluation_rejects_any_unknown_name(vsi_df):
    from TsT.benchmarks.vsi.benchmark import VSIBenchmark
    from TsT.evaluation import run_evaluation

    models = VSIBenchmark().get_feature_based_models()
    with pytest.raises(ValueError, match="Unknown question type"):
        run_evaluation(models, vsi_df, n_splits=3, question_types=["object_counting", "typo"], show_progress=False)


def test_locked_versions_match_the_lock_files():
    """The CLI's version warning compares against LOCKED_VERSIONS; keep it equal to uv.lock / constraints.txt."""
    import re
    from pathlib import Path

    from TsT.evaluation import LOCKED_VERSIONS

    root = Path(__file__).resolve().parents[2]
    lock = (root / "uv.lock").read_text()
    constraints = (root / "constraints.txt").read_text()
    for pkg, ver in LOCKED_VERSIONS.items():
        assert re.search(rf'^name = "{re.escape(pkg)}"\nversion = "{re.escape(ver)}"$', lock, re.M), pkg
        assert re.search(rf"^{re.escape(pkg)}=={re.escape(ver)}\b", constraints, re.M), pkg
