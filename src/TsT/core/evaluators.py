"""
Model evaluators for different bias detection approaches.

This module contains evaluator classes that implement the actual evaluation
logic for different model types while working with the unified cross-validation framework.
"""

import pandas as pd

from .protocols import ModelEvaluator, FeatureBasedBiasModel


class RandomForestEvaluator(ModelEvaluator):
    """Evaluator for feature-based Random Forest models"""

    def evaluate_fold(
        self,
        model: FeatureBasedBiasModel,  # Feature-based model
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_num: int,
        seed: int,
    ) -> float:
        """Evaluate RF on a single fold"""
        # Import here to avoid circular imports
        from ..evaluation import encode_categoricals, _make_estimator, _score

        # Feature engineering - this is RF-specific
        model.fit_feature_maps(train_df)
        train_features = model.add_features(train_df)
        test_features = model.add_features(test_df)

        # Prepare data for sklearn
        X_tr = train_features[model.feature_cols].copy()
        X_te = test_features[model.feature_cols].copy()
        encode_categoricals(X_tr, X_te)
        y_tr = train_features[target_col]
        y_te = test_features[target_col]

        # Train and evaluate
        estimator = _make_estimator(model.task, seed)
        estimator.fit(X_tr, y_tr)
        score = _score(estimator, X_te, y_te, model.metric)

        return score
