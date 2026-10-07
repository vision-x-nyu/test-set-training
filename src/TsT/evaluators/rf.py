"""
Random Forest evaluator for the unified evaluation framework.

This module contains the RandomForestEvaluator and related utilities for
training and evaluating Random Forest models on bias detection tasks.
"""

from typing import Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from ..core.protocols import ModelEvaluator, FeatureBasedBiasModel
from ..core.protocols import FoldResult, EvaluationResult
from ..utils import mean_relative_accuracy


def make_rf_estimator(task, seed):
    if task == "clf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    else:
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """Label-encode *object* columns (fit on **train only** to avoid leak).
    Unseen categories in test are mapped to -1."""
    cat_cols = X_train.select_dtypes(include="object").columns
    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        enc = LabelEncoder().fit(X_train[col].astype(str))
        mapping = {cls: i for i, cls in enumerate(enc.classes_)}
        X_train[col] = X_train[col].astype(str).map(mapping).astype(int)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)
        encoders[col] = enc
    return encoders


def score_rf(est, X, y, metric="acc") -> float:
    if metric == "acc":
        return float(est.score(X, y))  # plain accuracy
    elif metric == "mra":
        y_pred = est.predict(X)
        return mean_relative_accuracy(y_pred, y.values.astype(float))
    else:
        raise ValueError(f"Unknown metric: {metric}")


class RandomForestEvaluator(ModelEvaluator):
    """RF model evaluator for unified evaluation framework"""

    def train_and_evaluate_fold(
        self,
        model: FeatureBasedBiasModel,  # Feature-based model
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate RF on a single fold and return rich result"""

        # Feature engineering
        model.fit_feature_maps(train_df)
        train_features = model.add_features(train_df)
        test_features = model.add_features(test_df)

        # Prepare data
        X_tr = train_features[model.feature_cols].copy()
        X_te = test_features[model.feature_cols].copy()
        encode_categoricals(X_tr, X_te)
        y_tr = train_features[target_col]
        y_te = test_features[target_col]

        # Train and evaluate. Trees are fit in parallel; prediction sums the trees in a fixed order
        # (one job), so repeated runs give bit-identical scores.
        estimator = make_rf_estimator(model.task, seed)
        estimator.fit(X_tr, y_tr)
        estimator.set_params(n_jobs=1)
        score = score_rf(estimator, X_te, y_te, model.metric)

        # Per-sample scores s(x): P(gold) for classification, MRA of the prediction for regression
        if model.task == "clf":
            proba = estimator.predict_proba(X_te)
            classes_list = list(estimator.classes_)
            sample_predictions = {}
            for i, idx in enumerate(test_df.index):
                true_class = y_te.iloc[i]
                if true_class in classes_list:
                    col_idx = classes_list.index(true_class)
                    sample_predictions[idx] = float(proba[i, col_idx])
                else:
                    sample_predictions[idx] = 0.0
        else:
            y_pred = estimator.predict(X_te)
            sample_predictions = {
                idx: float(mean_relative_accuracy(float(y_pred[i]), float(y_te.iloc[i])))
                for i, idx in enumerate(test_df.index)
            }

        return FoldResult(
            fold_id=fold_id,
            score=score,
            train_size=len(train_df),
            test_idx=test_df.index.tolist(),
            metric=model.metric,
            metadata={
                "estimator_params": estimator.get_params(),
                "n_features": len(model.feature_cols),
            },
            sample_predictions=sample_predictions,
        )

    def process_results(
        self,
        model: FeatureBasedBiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Add feature importances to RF results"""

        # Train on full dataset for feature importances
        model.fit_feature_maps(df)
        full_df = model.add_features(df)
        X_full = full_df[model.feature_cols].copy()
        X_full_encoded = X_full.copy()
        encode_categoricals(X_full, X_full_encoded)
        y_full = full_df[target_col]

        # Feature importances are descriptive only (they do not enter any score); fixed seed.
        seed = 42
        rf_estimator = make_rf_estimator(model.task, seed)
        rf_estimator.fit(X_full_encoded, y_full)

        feature_importances = (
            pd.DataFrame({"feature": model.feature_cols, "importance": rf_estimator.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Update result
        evaluation_result.feature_importances = feature_importances
        evaluation_result.model_metadata.update(
            {
                "n_features": len(model.feature_cols),
                "feature_cols": model.feature_cols,
                "total_samples": len(df),
            }
        )

        return evaluation_result
