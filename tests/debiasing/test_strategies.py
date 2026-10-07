"""TopKDiversityStrategy: deterministic tie-breaking, group floor, missing scores."""

import numpy as np
import pandas as pd
import pytest

from TsT.debiasing.strategies import MissingScoreError, TopKDiversityStrategy


def _tied(n=1000, n_ties=303, seed=0, id_col=True):
    """Scores with many exact ties (like the 303 VSI-Bench NUM questions at s(x)=1.0)."""
    rng = np.random.RandomState(seed)
    s = rng.uniform(0, 0.9, n).round(2)  # few distinct values, like Random Forest probabilities
    s[rng.choice(n, n_ties, replace=False)] = 1.0
    df = pd.DataFrame({"question_type": rng.choice(list("abc"), n)}, index=np.arange(n))
    if id_col:
        df["id"] = rng.permutation(n) * 7 + 3  # ids unrelated to row position
    return df, pd.Series(s, index=df.index)


def test_ties_broken_by_id_and_invariant_to_row_order():
    df, s = _tied()
    strat = TopKDiversityStrategy("question_type", 10, id_col="id")
    a = strat.select_removal_candidates(df, s, 100)
    perm = np.random.RandomState(1).permutation(len(df))
    b = strat.select_removal_candidates(df.iloc[perm], s.iloc[perm], 100)
    assert a == b  # same questions in the same order
    # 100 of the 303 ties at 1.0: exactly the ones with the smallest ids
    tied = df.index[s == 1.0]
    assert set(a) == set(df.loc[tied].sort_values("id").index[:100])


def test_index_label_fallback_without_id_column():
    df, s = _tied(id_col=False)
    strat = TopKDiversityStrategy("question_type", 10)
    got = strat.select_removal_candidates(df, s, 100)
    assert got == list(s.sort_values(ascending=False, kind="stable").index[:100])
    perm = np.random.RandomState(2).permutation(len(df))
    assert strat.select_removal_candidates(df.iloc[perm], s.iloc[perm], 100) == got


def test_last_bit_noise_does_not_reorder_ties():
    df = pd.DataFrame({"question_type": ["a"] * 4, "id": [40, 30, 20, 10]})
    s = pd.Series([0.6866666666666668, 0.6866666666666665, 0.5, 0.6866666666666666], index=df.index)
    got = TopKDiversityStrategy("question_type", 0, id_col="id").select_removal_candidates(df, s, 3)
    assert got == [3, 1, 0]  # the three equal scores, by id


def test_legacy_sort_is_pandas_default():
    df, s = _tied()
    strat = TopKDiversityStrategy("question_type", 10, id_col="id", tie_break="legacy")
    assert strat.select_removal_candidates(df, s, 100) == list(s.sort_values(ascending=False).index[:100])


def test_group_floor_is_respected():
    df = pd.DataFrame({"question_type": ["a"] * 12 + ["b"] * 3, "id": range(15)})
    s = pd.Series(np.linspace(1, 0, 15), index=df.index)
    got = TopKDiversityStrategy("question_type", 10, id_col="id").select_removal_candidates(df, s, 15)
    assert got == [0, 1]  # "a" keeps 10; "b" (3 <= floor) loses nothing


def test_scores_may_cover_part_of_df_but_groups_count_all_rows():
    df = pd.DataFrame({"question_type": ["a"] * 6, "id": range(6)})
    s = pd.Series([0.9, 0.8], index=[0, 1])  # rows 2..5 are unscored (e.g. open-ended)
    got = TopKDiversityStrategy("question_type", 5, id_col="id").select_removal_candidates(df, s, 2)
    assert got == [0]  # only one removal keeps 5 rows in the group


def test_missing_score_raises():
    df = pd.DataFrame({"question_type": ["a", "a", "b"]}, index=[10, 11, 12])
    strat = TopKDiversityStrategy("question_type")
    with pytest.raises(MissingScoreError, match="1 of 3"):
        strat.compute_bias_scores(df, {10: 0.5, 11: 0.2})
    out = strat.compute_bias_scores(df, {10: 0.5, 11: 0.2, 12: 0.0, 99: 1.0})
    assert out.to_dict() == {10: 0.5, 11: 0.2, 12: 0.0}


def test_invalid_tie_break():
    with pytest.raises(ValueError):
        TopKDiversityStrategy("question_type", tie_break="random")


def test_benchmark_strategies_break_ties_by_their_id_column():
    from TsT.core.benchmark import BenchmarkRegistry

    for name, id_col in {
        "vsi": "id",
        "cvb": "idx",
        "mmmu": "id",
        "video_mme": "question_id",
        "mmstar": "index",
    }.items():
        bench = BenchmarkRegistry.get_benchmark(name)
        strat = bench.get_ibp_strategy()
        assert strat.id_col == id_col == bench.id_col
        assert strat.tie_break == "id"
