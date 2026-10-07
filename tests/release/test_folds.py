"""k-fold invariants for the TsT diagnostic: k disjoint folds, every question held out exactly once."""

import numpy as np
import pandas as pd
import pytest

from TsT.core.cross_validation import CrossValidationConfig, UnifiedCrossValidator
from TsT.core.protocols import FoldResult


class _Model:
    def __init__(self, task="clf"):
        self.name, self.format = f"m_{task}", ("mc" if task == "clf" else "num")
        self.task, self.metric = task, ("acc" if task == "clf" else "mra")
        self.feature_cols, self.target_col_override = ["f"], None

    def select_rows(self, df):
        return df


class _RecordingEvaluator:
    """Records every (train, test) split it is handed; scores each test row 1.0."""

    def __init__(self):
        self.splits = []

    def train_and_evaluate_fold(self, model, train_df, test_df, target_col, fold_id, seed):
        self.splits.append((set(train_df.index), set(test_df.index)))
        return FoldResult(
            fold_id=fold_id,
            score=1.0,
            train_size=len(train_df),
            test_idx=list(test_df.index),
            metric=model.metric,
            sample_predictions={i: 1.0 for i in test_df.index},
        )


def _df(n=103, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "gt_idx": rng.randint(0, 4, n),
            "ground_truth": rng.uniform(1, 9, n),
            "scene": rng.randint(0, 17, n),
            "f": rng.rand(n),
        },
        index=rng.permutation(10_000)[:n],
    )  # non-contiguous index on purpose


def _splits(task, df, target, k=5, seed=42, **cfg):
    cv = UnifiedCrossValidator(
        CrossValidationConfig(n_folds=k, random_state=seed, show_progress=False, verbose=False, **cfg)
    )
    ev = _RecordingEvaluator()
    rep = cv.run_cross_validation(_Model(task), ev, df, target, repeat_id=0)
    return ev.splits, rep


@pytest.mark.parametrize("group_col", [None, "scene"])
@pytest.mark.parametrize("task,target", [("clf", "gt_idx"), ("reg", "ground_truth")])
def test_every_row_held_out_exactly_once_and_train_test_disjoint(task, target, group_col):
    df = _df(300)
    splits, rep = _splits(task, df, target, group_col=group_col)
    assert len(splits) == 5
    tests = [t for _, t in splits]
    assert all(tr.isdisjoint(te) for tr, te in splits)
    assert all(tr | te == set(df.index) for tr, te in splits)
    assert sorted(i for t in tests for i in t) == sorted(df.index)  # exactly once
    scored = [i for f in rep.fold_results for i in f.sample_predictions]
    assert sorted(scored) == sorted(df.index)


def test_folds_deterministic_for_seed():
    df = _df()
    assert _splits("clf", df, "gt_idx")[0] == _splits("clf", df, "gt_idx")[0]
    assert _splits("clf", df, "gt_idx", seed=43)[0] != _splits("clf", df, "gt_idx")[0]


def test_stratification_preserves_label_proportions():
    df = _df(400)
    for _, te in _splits("clf", df, "gt_idx")[0]:
        p = df.loc[list(te), "gt_idx"].value_counts(normalize=True)
        q = df["gt_idx"].value_counts(normalize=True)
        assert (p - q).abs().max() < 0.05


@pytest.mark.parametrize("task,target", [("clf", "gt_idx"), ("reg", "ground_truth")])
def test_grouped_folds_never_split_a_group(task, target):
    df = _df(300)
    for tr, te in _splits(task, df, target, group_col="scene")[0]:
        assert set(df.loc[list(tr), "scene"]).isdisjoint(df.loc[list(te), "scene"])


def test_missing_group_col_raises():
    with pytest.raises(ValueError, match="group_col 'nope'"):
        _splits("clf", _df(), "gt_idx", group_col="nope")


def test_fold_seed_moves_only_the_partition():
    df = _df()
    seen = []

    class _SeedEvaluator(_RecordingEvaluator):
        def train_and_evaluate_fold(self, model, train_df, test_df, target_col, fold_id, seed):
            seen.append(seed)
            return super().train_and_evaluate_fold(model, train_df, test_df, target_col, fold_id, seed)

    def run(**cfg):
        cv = UnifiedCrossValidator(
            CrossValidationConfig(n_folds=5, random_state=42, show_progress=False, verbose=False, **cfg)
        )
        ev = _SeedEvaluator()
        cv.run_cross_validation(_Model("clf"), ev, df, "gt_idx", repeat_id=0)
        return ev.splits

    assert run(fold_seed=42) == run()  # fold_seed equal to random_state: the default partition
    seen.clear()
    assert run(fold_seed=7) != run()
    assert set(seen) == {42}  # the forest seed is unchanged
