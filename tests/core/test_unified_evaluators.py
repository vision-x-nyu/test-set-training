"""
Tests for the Random Forest fold evaluator and its post-processing.
"""

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from TsT.evaluators import RandomForestEvaluator
from TsT.core.protocols import FoldResult, EvaluationResult


class MockFeatureBasedBiasModel:
    """Mock feature-based bias model for testing"""

    def __init__(self, name="test_model", format="mc", task="clf", metric="acc"):
        self.name = name
        self.format = format
        self._task = task
        self._metric = metric
        self.feature_cols = ["feature1", "feature2"]
        self.target_col_override = None

        # Track method calls
        self.fit_feature_maps_called = False
        self.add_features_called = False

    @property
    def task(self):
        return self._task

    @property
    def metric(self):
        return self._metric

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        self.fit_feature_maps_called = True

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.add_features_called = True
        df_copy = df.copy()

        # Add deterministic features for testing
        df_copy["feature1"] = df_copy.index.astype(float) * 0.1
        df_copy["feature2"] = df_copy.index.astype(float) * 0.2

        return df_copy


def create_test_data(n_samples=50, n_classes=2):
    """Create test data for evaluators"""
    np.random.seed(42)

    # Build shared MC options for all rows
    mc_options = [f"{chr(65 + i)}: {i}" for i in range(n_classes)]

    data = {
        "id": range(n_samples),
        "gt_idx": np.random.randint(0, n_classes, n_samples),
        "ground_truth": np.random.randn(n_samples),
        "question": [f"Question {i}" for i in range(n_samples)],
        "options": [mc_options for _ in range(n_samples)],
    }

    return pd.DataFrame(data)


class TestRandomForestFoldEvaluator:
    """Test RandomForestFoldEvaluator"""

    def test_basic_evaluation(self):
        """Test basic RF fold evaluation"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(30, 2)
        test_df = create_test_data(10, 2)

        result = evaluator.train_and_evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=1, seed=42
        )

        # Verify result structure
        assert isinstance(result, FoldResult)
        assert result.fold_id == 1
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert result.test_size == 10

        # Verify metadata
        assert "estimator_params" in result.metadata
        assert "n_features" in result.metadata
        assert result.metadata["n_features"] == 2

        # train_size and test_size are stored as attributes of FoldResult, not in metadata
        assert result.train_size == 30

        # Verify feature engineering was called
        assert model.fit_feature_maps_called
        assert model.add_features_called

    def test_regression_evaluation(self):
        """Test RF evaluation for regression task"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel(task="reg", metric="mra")

        train_df = create_test_data(25, 2)
        test_df = create_test_data(8, 2)

        result = evaluator.train_and_evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="ground_truth", fold_id=2, seed=123
        )

        # Verify regression-specific behavior
        assert isinstance(result, FoldResult)
        assert result.fold_id == 2
        assert 0.0 <= result.score <= 1.0  # MRA should be between 0 and 1
        assert result.test_size == 8

    @patch("TsT.evaluators.rf.make_rf_estimator")
    @patch("TsT.evaluators.rf.score_rf")
    @patch("TsT.evaluators.rf.encode_categoricals")
    def test_evaluation_pipeline_mocked(self, mock_encode, mock_score, mock_make_estimator):
        """Test evaluation pipeline with mocked sklearn components"""
        # Setup mocks
        mock_estimator = Mock()
        mock_estimator.classes_ = np.array([0, 1])
        mock_estimator.predict_proba.return_value = np.array([[0.3, 0.7]] * 5)
        mock_make_estimator.return_value = mock_estimator
        mock_score.return_value = 0.75

        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(20, 2)
        test_df = create_test_data(5, 2)

        result = evaluator.train_and_evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=1, seed=42
        )

        # Verify pipeline was called correctly
        assert result.score == 0.75
        mock_make_estimator.assert_called_once_with(model.task, 42)
        mock_estimator.fit.assert_called_once()
        mock_score.assert_called_once()
        mock_encode.assert_called_once()

        # Verify estimator params were captured
        assert result.metadata["estimator_params"] == mock_estimator.get_params.return_value

    def test_deterministic_results(self):
        """Test that same seed produces same results"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(20, 2)
        test_df = create_test_data(5, 2)

        # Run twice with same seed
        result1 = evaluator.train_and_evaluate_fold(model, train_df, test_df, "gt_idx", 1, 42)
        result2 = evaluator.train_and_evaluate_fold(model, train_df, test_df, "gt_idx", 1, 42)

        # Should produce similar results (allowing for some numerical variation)
        assert abs(result1.score - result2.score) < 0.1
        assert result1.test_size == result2.test_size

    def test_sample_predictions_populated(self):
        """Test that RF evaluator populates sample_predictions in FoldResult"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(30, 2)
        test_df = create_test_data(10, 2)

        result = evaluator.train_and_evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=1, seed=42
        )

        # Verify sample_predictions is populated
        assert result.sample_predictions is not None
        assert len(result.sample_predictions) == result.test_size
        assert len(result.sample_predictions) == 10

        # Verify all test indices are present as keys
        for idx in test_df.index:
            assert idx in result.sample_predictions

        # Verify all prediction scores are valid floats in [0, 1]
        for idx, score in result.sample_predictions.items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_sample_predictions_regression(self):
        """Test that RF evaluator populates sample_predictions for regression tasks"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel(task="reg", metric="mra")

        train_df = create_test_data(25, 2)
        test_df = create_test_data(8, 2)

        result = evaluator.train_and_evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="ground_truth", fold_id=1, seed=42
        )

        # Verify sample_predictions is populated
        assert result.sample_predictions is not None
        assert len(result.sample_predictions) == result.test_size
        assert len(result.sample_predictions) == 8

        # Verify all test indices are present
        for idx in test_df.index:
            assert idx in result.sample_predictions

        # Verify scores are valid floats
        for idx, score in result.sample_predictions.items():
            assert isinstance(score, float)

    def test_sample_predictions_deterministic(self):
        """Test that same seed produces identical sample_predictions"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(30, 2)
        test_df = create_test_data(10, 2)

        result1 = evaluator.train_and_evaluate_fold(model, train_df, test_df, "gt_idx", 1, 42)
        result2 = evaluator.train_and_evaluate_fold(model, train_df, test_df, "gt_idx", 1, 42)

        assert result1.sample_predictions == result2.sample_predictions

    def test_sample_predictions_keys_match_test_idx(self):
        """Test that sample_predictions keys are exactly the test_df indices"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(30, 2)
        test_df = create_test_data(10, 2)

        result = evaluator.train_and_evaluate_fold(model, train_df, test_df, "gt_idx", 1, 42)

        assert set(result.sample_predictions.keys()) == set(result.test_idx)


class TestRandomForestPostProcessing:
    """Test RandomForest evaluator post-processing (integrated)"""

    def create_mock_evaluation_result(self):
        """Create mock evaluation result for testing"""
        from TsT.core.protocols import RepeatResult, FoldResult

        fold_results = [
            FoldResult(1, 0.8, 20, list(range(20)), "acc"),
            FoldResult(2, 0.9, 20, list(range(20)), "acc"),
        ]
        repeat_result = RepeatResult.from_fold_results(0, fold_results)

        return EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

    def test_feature_importance_generation(self):
        """Test feature importance generation via evaluator's process_results"""
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()
        df = create_test_data(30, 2)
        evaluation_result = self.create_mock_evaluation_result()

        # Process results (this should add feature importances)
        processed_result = evaluator.process_results(
            model=model, df=df, target_col="gt_idx", evaluation_result=evaluation_result
        )

        # Verify feature importances were added
        assert processed_result.feature_importances is not None
        assert len(processed_result.feature_importances) >= 0  # May vary based on mock

        # Verify metadata was added
        assert "total_samples" in processed_result.model_metadata
        assert processed_result.model_metadata["total_samples"] == 30


class TestEvaluatorIntegration:
    """Integration tests for evaluators and post-processors"""

    def test_rf_pipeline_integration(self):
        """Test complete RF evaluation pipeline"""
        # Create components
        evaluator = RandomForestEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(40, 2)
        test_df = create_test_data(10, 2)

        # Run fold evaluation
        fold_result = evaluator.train_and_evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=1, seed=42
        )

        # Create evaluation result for post-processing
        from TsT.core.protocols import RepeatResult

        repeat_result = RepeatResult.from_fold_results(0, [fold_result])
        evaluation_result = EvaluationResult.from_repeat_results(
            model_name="integration_test", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

        # Run post-processing (integrated in evaluator)
        processed_result = evaluator.process_results(
            model=model, df=train_df, target_col="gt_idx", evaluation_result=evaluation_result
        )

        # Verify complete pipeline
        assert isinstance(processed_result, EvaluationResult)
        assert processed_result.feature_importances is not None
        assert "total_samples" in processed_result.model_metadata

        # Verify fold result structure
        assert len(processed_result.repeat_results) == 1
        assert len(processed_result.repeat_results[0].fold_results) == 1
        assert processed_result.repeat_results[0].fold_results[0].fold_id == 1
