"""CV-Bench adapter: registration, pinned revision, preprocessing and an offline end-to-end run."""

import pytest

from TsT.benchmarks.cvb import benchmark
from TsT.benchmarks.cvb.benchmark import CVB_REVISION, CVBBenchmark
from TsT.core.benchmark import BenchmarkRegistry
from TsT.evaluation import evaluate_benchmark

QUESTION_TYPES = ["count_2d", "relation_2d", "depth_3d", "distance_3d"]


def test_registered_and_exported():
    assert "cvb" in BenchmarkRegistry.list_benchmarks()
    assert isinstance(BenchmarkRegistry.get_benchmark("cvb"), CVBBenchmark)
    assert isinstance(benchmark, CVBBenchmark)
    assert BenchmarkRegistry.get_benchmark("cvb") is not BenchmarkRegistry.get_benchmark("cvb")


def test_metadata_and_pins():
    meta = CVBBenchmark().get_metadata()
    assert meta["hf_repo"] == "nyu-visionx/CV-Bench"
    assert meta["default_revision"] == CVB_REVISION and len(CVB_REVISION) == 40
    assert meta["question_types"] == QUESTION_TYPES
    assert meta["formats"] == {"mc": QUESTION_TYPES}
    assert CVBBenchmark.id_col == "idx" and CVBBenchmark.group_col_hint == "image_id"


def test_every_model_targets_the_gold_option_index():
    for model in CVBBenchmark().get_feature_based_models():
        assert model.default_target_col == "gt_idx"
        assert model.target_col_override is None


def test_preprocess(cvb_df):
    assert set(cvb_df["question_type"]) == set(QUESTION_TYPES)
    assert (cvb_df["gt_idx"] == cvb_df["answer"].str[1].map(ord) - ord("A")).all()
    assert (cvb_df["n_options"] == cvb_df["choices"].map(len)).all()
    assert cvb_df["image_id"].str.contains("/").all()


def test_models_select_every_question_of_their_type(cvb_df):
    for model in CVBBenchmark().get_feature_based_models():
        assert len(model.select_rows(cvb_df)) == (cvb_df["question_type"] == model.name).sum(), model.name


@pytest.mark.parametrize("group_col", [None, "image_id"])
def test_end_to_end_without_target_col(cvb_df, group_col):
    """No target_col given: every model uses gt_idx (as the reported numbers do)."""
    run = evaluate_benchmark("cvb", df=cvb_df, n_splits=3, group_col=group_col, show_progress=False)
    assert not run.errors
    assert [r.model_name for r in run.results] == QUESTION_TYPES
    assert sum(r.count for r in run.results) == len(cvb_df)
    assert {p["target_col"] for p in run.summary()["per_question_type"]} == {"gt_idx"}


def test_qa_models():
    models = CVBBenchmark().get_qa_models()
    assert [m.name for m in models] == ["cvb_mc"]
    assert CVBBenchmark().supports_rf and CVBBenchmark().supports_llm
