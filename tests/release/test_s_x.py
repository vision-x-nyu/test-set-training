"""Per-question s(x) export: one row per question, joinable on the benchmark id."""

import numpy as np
import pytest

from TsT.evaluation import evaluate_benchmark


@pytest.fixture(scope="module")
def vsi_run():
    from TsT.benchmarks.vsi.benchmark import VSIBenchmark

    from ..conftest import load_fixture

    df = VSIBenchmark.preprocess(load_fixture("vsi_test_subset.jsonl", "options"))
    return evaluate_benchmark("vsi", df=df, n_splits=3, repeats=2, show_progress=False)


def test_one_row_per_question_and_ids_join_one_to_one(vsi_run, vsi_df):
    s_x = vsi_run.s_x()
    assert s_x["id"].is_unique
    assert set(s_x["id"]) == set(vsi_df["id"])
    assert (s_x["n_repeats"] == 2).all()


def test_chance_columns(vsi_run, vsi_df):
    s_x = vsi_run.s_x().merge(vsi_df[["id", "options"]], on="id")
    mc, num = s_x[s_x["format"] == "mc"], s_x[s_x["format"] == "num"]
    assert (mc["n_options"] == mc["options"].map(len)).all()
    assert np.allclose(mc["chance"], 1.0 / mc["n_options"].astype(float))
    assert np.allclose(mc["s_minus_chance"], mc["s"] - mc["chance"])
    assert num["n_options"].isna().all() and num["chance"].isna().all() and num["s_minus_chance"].isna().all()
    assert s_x["s"].between(0, 1).all()
    assert set(mc["metric"]) == {"acc"} and set(num["metric"]) == {"mra"}


def test_mc_s_averages_to_mean_gold_probability(vsi_run):
    """The fold score is accuracy, s(x) is P(gold): related but not equal; s must stay a probability."""
    s_x = vsi_run.s_x()
    for name, g in s_x[s_x["format"] == "mc"].groupby("model"):
        assert 0 < g["s"].mean() < 1, name


def test_cvb_n_options_from_choices(cvb_df):
    run = evaluate_benchmark("cvb", df=cvb_df, n_splits=3, show_progress=False)
    s_x = run.s_x().merge(cvb_df[["idx", "choices"]], on="idx")
    assert s_x["idx"].is_unique and len(s_x) == len(cvb_df)
    assert (s_x["n_options"] == s_x["choices"].map(len)).all()


def test_summary_records_the_id_column(cvb_df):
    from TsT.evaluation import evaluate_benchmark

    run = evaluate_benchmark("cvb", df=cvb_df, n_splits=3, show_progress=False)
    summary = run.summary()
    assert summary["id_col"] == "idx" and "idx" in run.s_x().columns
    assert "differs_from_lock" in summary["environment"]
