"""
Model evaluators for different bias detection approaches.

This module contains evaluator classes that implement the actual evaluation
logic for different model types while working with the unified cross-validation framework.
"""

import pandas as pd
from typing import Dict, Any

from .protocols import ModelEvaluator, FeatureBasedBiasModel
from .cross_validation import FoldEvaluator, PostProcessor
from .results import FoldResult, EvaluationResult


class RandomForestEvaluator(ModelEvaluator):
    """Legacy evaluator for feature-based Random Forest models"""

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


class RandomForestFoldEvaluator(FoldEvaluator):
    """RF-specific fold evaluator for unified evaluation framework"""

    def evaluate_fold(
        self,
        model: FeatureBasedBiasModel,  # Feature-based model
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate RF on a single fold and return rich result"""
        # Import here to avoid circular imports
        from ..evaluation import encode_categoricals, _make_estimator, _score

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

        # Train and evaluate
        estimator = _make_estimator(model.task, seed)
        estimator.fit(X_tr, y_tr)
        score = _score(estimator, X_te, y_te, model.metric)

        return FoldResult(
            fold_id=fold_id,
            score=score,
            fold_size=len(test_df),
            metadata={
                "estimator_params": estimator.get_params(),
                "n_features": len(model.feature_cols),
                "train_size": len(train_df),
            },
        )


class RandomForestPostProcessor(PostProcessor):
    """Generate feature importances for RF models"""

    def process_results(
        self,
        model: FeatureBasedBiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Add feature importances to RF results"""
        # Import here to avoid circular imports
        from ..evaluation import encode_categoricals, _make_estimator

        # Train on full dataset for feature importances
        model.fit_feature_maps(df)
        full_df = model.add_features(df)
        X_full = full_df[model.feature_cols].copy()
        X_full_encoded = X_full.copy()
        encode_categoricals(X_full, X_full_encoded)
        y_full = full_df[target_col]

        # Use first repeat's first fold's seed for consistency
        seed = 42  # Default seed - could be made configurable
        if evaluation_result.repeat_results and evaluation_result.repeat_results[0].fold_results:
            # Try to extract seed from metadata if available
            pass

        estimator = _make_estimator(model.task, seed)
        estimator.fit(X_full_encoded, y_full)

        feature_importances = (
            pd.DataFrame({"feature": model.feature_cols, "importance": estimator.feature_importances_})
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


class LLMFoldEvaluator(FoldEvaluator):
    """LLM-specific fold evaluator for unified evaluation framework"""

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.trainable_predictor = None

    def evaluate_fold(
        self,
        model,  # BiasModel
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate LLM on a single fold"""
        # For now, implement a placeholder that uses the legacy LLM evaluation
        # This will be replaced with production LLM infrastructure
        from .llm_evaluators import TemporaryLLMEvaluator

        temp_evaluator = TemporaryLLMEvaluator(self.llm_config)
        score = temp_evaluator.evaluate_fold(model, train_df, test_df, target_col, fold_id, seed)

        return FoldResult(
            fold_id=fold_id,
            score=score,
            fold_size=len(test_df),
            metadata={
                "training_size": len(train_df),
                "model_name": self.llm_config.get("model_name", "unknown"),
                "llm_config": self.llm_config,
            },
        )


class LLMPostProcessor(PostProcessor):
    """Generate LLM-specific metadata"""

    def __init__(self, llm_config: Dict[str, Any], zero_shot_baseline: float = None):
        self.llm_config = llm_config
        self.zero_shot_baseline = zero_shot_baseline

    def process_results(
        self,
        model,  # BiasModel
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Add LLM-specific metadata"""
        # Calculate improvement over zero-shot
        improvement = evaluation_result.overall_mean - self.zero_shot_baseline if self.zero_shot_baseline else 0.0

        # Mock feature importances for compatibility
        feature_importances = pd.DataFrame(
            {
                "feature": ["llm_finetuning", "zero_shot_baseline", "improvement"],
                "importance": [evaluation_result.overall_mean, self.zero_shot_baseline or 0.0, improvement],
            }
        )

        evaluation_result.feature_importances = feature_importances
        evaluation_result.model_metadata.update(
            {
                "zero_shot_baseline": self.zero_shot_baseline,
                "improvement": improvement,
                "llm_config": self.llm_config,
                "total_samples": len(df),
            }
        )

        return evaluation_result
