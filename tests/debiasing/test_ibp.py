"""IBP loop and score aggregation (raw s and delta), with synthetic TsT results and on the offline fixture."""

import pandas as pd
import pytest

from TsT.core.protocols import EvaluationResult, FoldResult, RepeatResult
from TsT.debiasing.config import IBPConfig
from TsT.debiasing.ibp import aggregate_sample_predictions, debias_benchmark, iterative_bias_pruning
from TsT.debiasing.strategies import MissingScoreError, TopKDiversityStrategy


def _result(preds, zero_shot=None, name="m"):
    folds = [
        FoldResult(
            fold_id=1, score=0.0, train_size=1, test_idx=list(preds), metric="acc", sample_predictions=dict(preds)
        )
    ]
    res = EvaluationResult.from_repeat_results(name, "mc", "acc", [RepeatResult.from_fold_results(0, folds)])
    res.zero_shot_sample_scores = zero_shot
    return res


def test_raw_scores_average_over_repeats():
    r = _result({0: 0.2, 1: 0.9})
    r.repeat_results.append(
        RepeatResult.from_fold_results(1, [FoldResult(2, 0.0, 1, [0, 1], "acc", sample_predictions={0: 0.4, 1: 0.7})])
    )
    assert aggregate_sample_predictions([r]) == pytest.approx({0: 0.3, 1: 0.8})


def test_delta_subtracts_zero_shot():
    r = _result({0: 0.9, 1: 0.6, 2: 0.3}, zero_shot={0: 0.95, 1: 0.1, 2: 0.3})
    assert aggregate_sample_predictions([r], score="delta") == pytest.approx({0: -0.05, 1: 0.5, 2: 0.0})
    # raw ranking would remove question 0 first; delta ranks question 1 first
    df = pd.DataFrame({"question_type": ["a"] * 3, "id": ["x", "y", "z"]})
    strat = TopKDiversityStrategy("question_type", 0, id_col="id")
    for score, first in (("s", 0), ("delta", 1)):
        _, res = iterative_bias_pruning(
            df, strat, IBPConfig(budget=1, batch_size=1, score=score), lambda d: [r], id_col="id"
        )
        assert res.removed_indices == [first]


def test_delta_needs_zero_shot_scores():
    with pytest.raises(ValueError, match="zero-shot"):
        aggregate_sample_predictions([_result({0: 0.5})], score="delta")
    with pytest.raises(MissingScoreError):
        aggregate_sample_predictions([_result({0: 0.5, 1: 0.5}, zero_shot={0: 0.1})], score="delta")


def test_failed_result_raises():
    r = _result({0: 0.5})
    r.model_metadata = {"error": "boom"}
    with pytest.raises(ValueError, match="boom"):
        aggregate_sample_predictions([r])


def _fake_tst(scores):
    """run_tst_fn returning fixed per-row scores for the rows still present."""
    return lambda d: [_result({i: scores[i] for i in d.index if i in scores})]


def test_loop_budget_batches_and_ids():
    df = pd.DataFrame({"question_type": ["a"] * 20, "id": [f"q{i:02d}" for i in range(20)]})
    scores = {i: 1.0 - i / 20 for i in range(20)}
    kept, res = iterative_bias_pruning(
        df,
        TopKDiversityStrategy("question_type", 5, id_col="id"),
        IBPConfig(budget=7, batch_size=3),
        _fake_tst(scores),
        id_col="id",
    )
    assert [len(it.removed_ids) for it in res.iterations] == [3, 3, 1]
    assert res.removed_ids == [f"q{i:02d}" for i in range(7)] and len(kept) == 13
    assert res.iterations[0].mean_bias_score == pytest.approx(sum(scores.values()) / 20)
    assert res.iterations[1].dataset_size_before == 17 and res.stop_reason == "budget"


def test_early_stop_at_or_below_threshold():
    df = pd.DataFrame({"question_type": ["a"] * 10, "id": range(10)})
    scores = {i: 0.5 if i < 2 else 0.1 for i in range(10)}
    _, res = iterative_bias_pruning(
        df,
        TopKDiversityStrategy("question_type", 0, id_col="id"),
        IBPConfig(budget=6, batch_size=2, early_stop_threshold=0.1),
        _fake_tst(scores),
        id_col="id",
    )
    assert res.removed_ids == [0, 1] and res.stop_reason == "early_stop"  # max 0.1 <= 0.1 stops


def test_unranked_rows_are_kept_and_must_not_lack_scores():
    df = pd.DataFrame({"question_type": ["a"] * 6, "question_format": ["mc"] * 4 + ["oe"] * 2, "id": range(6)})
    scores = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}  # the open-ended rows are not scored
    ranked = lambda d: d.index[d["question_format"] == "mc"]  # noqa: E731
    strat = TopKDiversityStrategy("question_type", 0, id_col="id")
    _, res = iterative_bias_pruning(
        df, strat, IBPConfig(budget=4, batch_size=4), _fake_tst(scores), ranked_index_fn=ranked, id_col="id"
    )
    assert sorted(res.removed_ids) == [0, 1, 2, 3]
    with pytest.raises(MissingScoreError):  # without the ranked set, unscored rows are an error
        iterative_bias_pruning(df, strat, IBPConfig(budget=1, batch_size=1), _fake_tst(scores), id_col="id")


def test_config_validation():
    for kwargs in (dict(budget=0), dict(budget=5, batch_size=6), dict(budget=5, score="raw")):
        with pytest.raises(ValueError):
            IBPConfig(**kwargs)


def test_debias_benchmark_on_fixture(vsi_df):
    run = debias_benchmark("vsi", alloc="per_format", budget=20, batch_size=5, n_splits=3, df=vsi_df)
    s = run.summary()
    assert run.n_removed == 20 and len(run.kept_ids) == len(vsi_df) - 20
    assert set(run.removed_ids) | set(run.kept_ids) == set(vsi_df["id"]) and not set(run.removed_ids) & set(
        run.kept_ids
    )
    # 180 MC and 120 NUM fixture questions
    assert {g["group"]: (g["budget"], g["n_iterations"]) for g in s["groups"]} == {"mc": (12, 3), "num": (8, 2)}
    assert s["revision"].startswith("bc96b17") and s["tie_break"] == "id" and s["score"] == "s"
    # same inputs, same removals (and independent of row order)
    again = debias_benchmark("vsi", alloc="per_format", budget=20, batch_size=5, n_splits=3, df=vsi_df)
    assert again.removed_ids == run.removed_ids


def test_debias_benchmark_rejects_delta_for_rf(vsi_df):
    with pytest.raises(ValueError, match="TsT-LLM"):
        debias_benchmark("vsi", alloc="per_format", budget=10, score="delta", df=vsi_df)
