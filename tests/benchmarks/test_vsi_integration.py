"""VSI-Bench adapter: registration, pinned revision, preprocessing and an offline end-to-end run."""

import pytest

from TsT.benchmarks.vsi import benchmark
from TsT.benchmarks.vsi.benchmark import VSI_REVISION, VSIBenchmark
from TsT.core.benchmark import BenchmarkRegistry
from TsT.evaluation import evaluate_benchmark

NUM = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"]
MC = ["object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"]


def test_registered_and_exported():
    assert "vsi" in BenchmarkRegistry.list_benchmarks()
    assert isinstance(BenchmarkRegistry.get_benchmark("vsi"), VSIBenchmark)
    assert isinstance(benchmark, VSIBenchmark)


def test_metadata_and_pins():
    meta = VSIBenchmark().get_metadata()
    assert meta["hf_repo"] == "nyu-visionx/VSI-Bench"
    assert meta["default_revision"] == VSI_REVISION and VSI_REVISION.startswith("bc96b17")
    assert meta["feature_sets"] == ["default", "paper"]
    assert meta["formats"] == {"num": NUM, "mc": MC}
    assert VSIBenchmark.id_col == "id" and VSIBenchmark.group_col_hint == "scene_name"


def test_preprocess(vsi_df):
    mc = vsi_df[vsi_df["question_format"] == "mc"]
    num = vsi_df[vsi_df["question_format"] == "num"]
    assert (num["gt_idx"] == -1).all() and (num["gt_val"] == num["ground_truth"]).all()
    assert (mc["gt_idx"] == mc["ground_truth"].map("ABCD".index)).all()
    assert all(opts[i].endswith(val) for opts, i, val in zip(mc["options"], mc["gt_idx"], mc["gt_val"]))


def test_models_select_every_question_of_their_type(vsi_df):
    for model in VSIBenchmark().get_feature_based_models():
        n_type = vsi_df["question_type"].str.startswith(model.name).sum()
        assert len(model.select_rows(vsi_df)) == n_type, model.name


@pytest.mark.parametrize("feature_set,group_col", [("default", None), ("paper", "scene_name")])
def test_end_to_end(vsi_df, feature_set, group_col):
    run = evaluate_benchmark(
        "vsi", df=vsi_df, n_splits=3, feature_set=feature_set, group_col=group_col, show_progress=False
    )
    assert not run.errors
    assert [r.model_name for r in run.results] == NUM + MC
    assert sum(r.count for r in run.results) == len(vsi_df)
    assert 0 < run.weighted_mean < 1


def test_qa_models():
    models = VSIBenchmark().get_qa_models()
    assert [m.name for m in models] == ["vsi_mc", "vsi_num"]
    assert VSIBenchmark().supports_rf and VSIBenchmark().supports_llm
