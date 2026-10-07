"""Pin the scoring definitions the reported numbers depend on."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from TsT import utils
from TsT.evaluators.rf import RandomForestEvaluator


def test_mra_uses_ten_thresholds_050_to_095():
    # relative error 0.30 passes threshold t iff 0.30 < 1 - t
    assert utils.mean_relative_accuracy(7.0, 10.0) == pytest.approx(
        np.mean([0.3 < 1 - t for t in np.linspace(0.5, 0.95, 10)])
    )


@pytest.mark.parametrize("pred,gt,tst,official", [(3, 4, 0.5, 0.6), (5, 4, 0.5, 0.6), (1, 2, 0.0, 0.1)])
def test_mra_boundary_is_strict(pred, gt, tst, official):
    """TsT counts rel_err < 1 - t; the official VSI-Bench scorer counts rel_err <= 1 - t.

    They differ only when the relative error lands exactly on a threshold (here 0.25 and 0.5).
    """
    assert utils.mean_relative_accuracy(pred, gt) == pytest.approx(tst)
    rel = abs(pred - gt) / gt
    assert np.mean([rel <= 1 - t + 1e-12 for t in np.linspace(0.5, 0.95, 10)]) == pytest.approx(official)


def test_mra_zero_target_conventions():
    assert utils.mean_relative_accuracy(0.0, 0.0) == 1.0
    assert utils.mean_relative_accuracy(1.0, 0.0) == 0.0


def test_no_multiple_choice_text_parser_ships():
    """TsT-RF never parses free text; the unseeded random-fallback MC parser is not part of this release."""
    assert not hasattr(utils, "parse_multi_choice_response")


class _MC:
    name, format, task, metric, target_col_override = "mc", "mc", "clf", "acc", None
    feature_cols = ["f1", "f2"]

    def fit_feature_maps(self, train_df):
        pass

    def add_features(self, df):
        return df


def test_mc_sample_score_is_probability_of_gold():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"f1": rng.rand(60), "f2": rng.rand(60), "gt_idx": rng.randint(0, 3, 60)}, index=range(100, 160))
    train, test = df.iloc[:45], df.iloc[45:]
    fold = RandomForestEvaluator().train_and_evaluate_fold(_MC(), train, test, "gt_idx", fold_id=1, seed=7)

    est = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1).fit(train[["f1", "f2"]], train["gt_idx"])
    proba = est.predict_proba(test[["f1", "f2"]])
    expected = {idx: proba[i, list(est.classes_).index(y)] for i, (idx, y) in enumerate(test["gt_idx"].items())}
    assert fold.sample_predictions.keys() == expected.keys()
    assert all(fold.sample_predictions[k] == pytest.approx(v) for k, v in expected.items())
    assert fold.score == pytest.approx(est.score(test[["f1", "f2"]], test["gt_idx"]))
