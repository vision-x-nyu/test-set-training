"""A model that crashes must never drop silently out of the headline mean."""

import json

import pandas as pd
import pytest

from TsT.core.benchmark import Benchmark, BenchmarkRegistry
from TsT.evaluation import ModelEvaluationError, run_evaluation, summarize_results


class _Good:
    name, format, task, metric, feature_cols, target_col_override = "good", "mc", "clf", "acc", ["f"], None

    def select_rows(self, df):
        return df

    def fit_feature_maps(self, train_df):
        pass

    def add_features(self, df):
        return df


class _Boom(_Good):
    name = "boom"

    def fit_feature_maps(self, train_df):
        raise RuntimeError("feature bug")


def _df():
    return pd.DataFrame({"id": range(40), "gt_idx": [0, 1] * 20, "f": [0, 1] * 20, "question_type": "t"})


def test_failed_model_raises_by_default():
    with pytest.raises(ModelEvaluationError, match="boom"):
        run_evaluation(question_models=[_Good(), _Boom()], df_full=_df(), n_splits=2, target_col="gt_idx")


def test_keep_going_flags_the_failure_and_excludes_it_from_the_means():
    res = run_evaluation(
        question_models=[_Good(), _Boom()], df_full=_df(), n_splits=2, target_col="gt_idx", keep_going=True
    )
    assert [r.model_name for r in res] == ["good", "boom"]
    assert "feature bug" in res[1].model_metadata["error"]
    summary = summarize_results(res)
    assert summary["complete"] is False and list(summary["errors"]) == ["boom"]
    assert summary["n_models"] == 1 and summary["total_count"] == 40
    assert summary["weighted_mean"] == summary["macro_mean"] == pytest.approx(1.0)


def _register_boom_benchmark():
    @BenchmarkRegistry.register
    class BoomBenchmark(Benchmark):
        name = "boom_bench"
        default_revision = "v0"

        def load_data(self, revision=None):
            return _df()

        def get_feature_based_models(self, feature_set="default"):
            return [_Good(), _Boom()]


def test_cli_crash_stops_the_run(clean_registry, capsys):
    from TsT.__main__ import main

    _register_boom_benchmark()
    assert main(["--benchmark", "boom_bench", "--n_splits", "2", "--target_col", "gt_idx"]) == 1
    captured = capsys.readouterr()
    assert "error: Evaluation failed for boom" in captured.err and "weighted mean" not in captured.out


def test_cli_keep_going_exits_nonzero_and_marks_summary_incomplete(clean_registry, tmp_path, capsys):
    from TsT.__main__ import main

    _register_boom_benchmark()
    code = main(["-b", "boom_bench", "-k", "2", "-t", "gt_idx", "--keep_going", "-o", str(tmp_path)])
    assert code == 1
    assert "boom" in capsys.readouterr().err
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"] is False and list(summary["errors"]) == ["boom"]
