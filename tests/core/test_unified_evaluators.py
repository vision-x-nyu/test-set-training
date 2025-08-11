"""
Tests for unified evaluation framework evaluators and post-processors.

These tests ensure the new fold evaluators and post-processors work correctly
with the unified evaluation framework and produce expected results.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from TsT.core.evaluators import RandomForestFoldEvaluator, RandomForestPostProcessor, LLMFoldEvaluator, LLMPostProcessor
from TsT.core.results import FoldResult, EvaluationResult


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

    data = {
        "id": range(n_samples),
        "gt_idx": np.random.randint(0, n_classes, n_samples),
        "ground_truth": np.random.randn(n_samples),
        "question": [f"Question {i}" for i in range(n_samples)],
    }

    return pd.DataFrame(data)


class TestRandomForestFoldEvaluator:
    """Test RandomForestFoldEvaluator"""

    def test_basic_evaluation(self):
        """Test basic RF fold evaluation"""
        evaluator = RandomForestFoldEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(30, 2)
        test_df = create_test_data(10, 2)

        result = evaluator.evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=1, seed=42
        )

        # Verify result structure
        assert isinstance(result, FoldResult)
        assert result.fold_id == 1
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert result.fold_size == 10

        # Verify metadata
        assert "estimator_params" in result.metadata
        assert "n_features" in result.metadata
        assert "train_size" in result.metadata
        assert result.metadata["train_size"] == 30
        assert result.metadata["n_features"] == 2

        # Verify feature engineering was called
        assert model.fit_feature_maps_called
        assert model.add_features_called

    def test_regression_evaluation(self):
        """Test RF evaluation for regression task"""
        evaluator = RandomForestFoldEvaluator()
        model = MockFeatureBasedBiasModel(task="reg", metric="mra")

        train_df = create_test_data(25, 2)
        test_df = create_test_data(8, 2)

        result = evaluator.evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="ground_truth", fold_id=2, seed=123
        )

        # Verify regression-specific behavior
        assert isinstance(result, FoldResult)
        assert result.fold_id == 2
        assert 0.0 <= result.score <= 1.0  # MRA should be between 0 and 1
        assert result.fold_size == 8

    @patch("TsT.evaluation._make_estimator")
    @patch("TsT.evaluation._score")
    @patch("TsT.evaluation.encode_categoricals")
    def test_evaluation_pipeline_mocked(self, mock_encode, mock_score, mock_make_estimator):
        """Test evaluation pipeline with mocked sklearn components"""
        # Setup mocks
        mock_estimator = Mock()
        mock_make_estimator.return_value = mock_estimator
        mock_score.return_value = 0.75

        evaluator = RandomForestFoldEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(20, 2)
        test_df = create_test_data(5, 2)

        result = evaluator.evaluate_fold(
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
        evaluator = RandomForestFoldEvaluator()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(20, 2)
        test_df = create_test_data(5, 2)

        # Run twice with same seed
        result1 = evaluator.evaluate_fold(model, train_df, test_df, "gt_idx", 1, 42)
        result2 = evaluator.evaluate_fold(model, train_df, test_df, "gt_idx", 1, 42)

        # Should produce similar results (allowing for some numerical variation)
        assert abs(result1.score - result2.score) < 0.1
        assert result1.fold_size == result2.fold_size


class TestRandomForestPostProcessor:
    """Test RandomForestPostProcessor"""

    def create_mock_evaluation_result(self):
        """Create mock evaluation result for testing"""
        from TsT.core.results import RepeatResult, FoldResult

        fold_results = [
            FoldResult(1, 0.8, 20),
            FoldResult(2, 0.9, 20),
        ]
        repeat_result = RepeatResult.from_fold_results(0, fold_results)

        return EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

    @patch("TsT.evaluation._make_estimator")
    @patch("TsT.evaluation.encode_categoricals")
    def test_feature_importance_generation(self, mock_encode, mock_make_estimator):
        """Test feature importance generation"""
        # Setup mock estimator with feature importances
        mock_estimator = Mock()
        mock_estimator.feature_importances_ = np.array([0.6, 0.4])
        mock_make_estimator.return_value = mock_estimator

        processor = RandomForestPostProcessor()
        model = MockFeatureBasedBiasModel()
        df = create_test_data(30, 2)
        evaluation_result = self.create_mock_evaluation_result()

        # Process results
        processed_result = processor.process_results(
            model=model, df=df, target_col="gt_idx", evaluation_result=evaluation_result
        )

        # Verify feature importances were added
        assert processed_result.feature_importances is not None
        assert len(processed_result.feature_importances) == 2
        assert list(processed_result.feature_importances["feature"]) == ["feature1", "feature2"]
        assert np.allclose(processed_result.feature_importances["importance"], [0.6, 0.4])

        # Verify importances are sorted by importance (descending)
        importances = processed_result.feature_importances["importance"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))

        # Verify metadata was added
        assert "n_features" in processed_result.model_metadata
        assert "feature_cols" in processed_result.model_metadata
        assert "total_samples" in processed_result.model_metadata
        assert processed_result.model_metadata["n_features"] == 2
        assert processed_result.model_metadata["total_samples"] == 30

        # Verify sklearn pipeline was called
        mock_make_estimator.assert_called_once()
        mock_estimator.fit.assert_called_once()
        mock_encode.assert_called_once()

    def test_metadata_preservation(self):
        """Test that existing metadata is preserved"""
        processor = RandomForestPostProcessor()
        model = MockFeatureBasedBiasModel()
        df = create_test_data(20, 2)

        evaluation_result = self.create_mock_evaluation_result()
        # Add some existing metadata
        evaluation_result.model_metadata["existing_key"] = "existing_value"

        with patch("TsT.evaluation._make_estimator") as mock_make_estimator:
            mock_estimator = Mock()
            mock_estimator.feature_importances_ = np.array([0.5, 0.5])
            mock_make_estimator.return_value = mock_estimator

            processed_result = processor.process_results(model, df, "gt_idx", evaluation_result)

        # Verify existing metadata was preserved
        assert processed_result.model_metadata["existing_key"] == "existing_value"
        # And new metadata was added
        assert "n_features" in processed_result.model_metadata


class TestLLMFoldEvaluator:
    """Test LLMFoldEvaluator"""

    def test_basic_initialization(self):
        """Test basic LLM evaluator initialization"""
        llm_config = {"model_name": "test/model", "batch_size": 4, "epochs": 1}

        evaluator = LLMFoldEvaluator(llm_config)

        assert evaluator.llm_config == llm_config
        assert evaluator.trainable_predictor is None

    @patch("TsT.evaluation.evaluate_bias_model_llm")
    def test_evaluation_with_legacy_system(self, mock_evaluate_llm):
        """Test LLM evaluation using legacy system"""
        # Setup mock to return (mean_score, std_score, feature_importances, count)
        mock_evaluate_llm.return_value = (0.85, 0.1, None, 20)

        llm_config = {"model_name": "test/model"}
        evaluator = LLMFoldEvaluator(llm_config)
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(15, 2)
        test_df = create_test_data(5, 2)

        result = evaluator.evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=3, seed=456
        )

        # Verify result structure
        assert isinstance(result, FoldResult)
        assert result.fold_id == 3
        assert result.score == 0.85
        assert result.fold_size == 5

        # Verify metadata
        assert "training_size" in result.metadata
        assert "model_name" in result.metadata
        assert "llm_config" in result.metadata
        assert result.metadata["training_size"] == 15
        assert result.metadata["model_name"] == "test/model"

        # Verify legacy function was called
        mock_evaluate_llm.assert_called_once()

    def test_config_handling(self):
        """Test LLM config handling"""
        # Test with minimal config
        evaluator1 = LLMFoldEvaluator({})
        assert evaluator1.llm_config == {}

        # Test with full config
        full_config = {"model_name": "google/gemma-2-2b-it", "batch_size": 8, "learning_rate": 2e-4, "epochs": 2}
        evaluator2 = LLMFoldEvaluator(full_config)
        assert evaluator2.llm_config == full_config


class TestLLMPostProcessor:
    """Test LLMPostProcessor"""

    def create_mock_evaluation_result(self):
        """Create mock evaluation result for testing"""
        from TsT.core.results import RepeatResult, FoldResult

        fold_results = [FoldResult(1, 0.75, 25)]
        repeat_result = RepeatResult.from_fold_results(0, fold_results)

        return EvaluationResult.from_repeat_results(
            model_name="llm_test", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

    def test_basic_processing(self):
        """Test basic LLM post-processing"""
        llm_config = {"model_name": "test/model"}
        zero_shot_baseline = 0.25

        processor = LLMPostProcessor(llm_config, zero_shot_baseline)
        model = MockFeatureBasedBiasModel()
        df = create_test_data(50, 2)
        evaluation_result = self.create_mock_evaluation_result()

        processed_result = processor.process_results(
            model=model, df=df, target_col="gt_idx", evaluation_result=evaluation_result
        )

        # Verify mock feature importances were added
        assert processed_result.feature_importances is not None
        assert len(processed_result.feature_importances) == 3

        expected_features = ["llm_finetuning", "zero_shot_baseline", "improvement"]
        assert list(processed_result.feature_importances["feature"]) == expected_features

        # Verify importance values
        importances = processed_result.feature_importances["importance"].values
        assert importances[0] == 0.75  # LLM score
        assert importances[1] == 0.25  # Zero-shot baseline
        assert importances[2] == 0.5  # Improvement (0.75 - 0.25)

        # Verify metadata
        assert "zero_shot_baseline" in processed_result.model_metadata
        assert "improvement" in processed_result.model_metadata
        assert "llm_config" in processed_result.model_metadata
        assert "total_samples" in processed_result.model_metadata

        assert processed_result.model_metadata["zero_shot_baseline"] == 0.25
        assert processed_result.model_metadata["improvement"] == 0.5
        assert processed_result.model_metadata["total_samples"] == 50

    def test_without_zero_shot_baseline(self):
        """Test processing without zero-shot baseline raises NotImplementedError"""
        llm_config = {"model_name": "test/model"}
        processor = LLMPostProcessor(llm_config, zero_shot_baseline=None)

        model = MockFeatureBasedBiasModel()
        df = create_test_data(30, 2)
        evaluation_result = self.create_mock_evaluation_result()

        # Should raise NotImplementedError when zero-shot baseline is None
        with pytest.raises(NotImplementedError, match="Zero-shot baseline is required"):
            processor.process_results(model, df, "gt_idx", evaluation_result)

    def test_metadata_preservation(self):
        """Test that existing metadata is preserved"""
        processor = LLMPostProcessor({}, 0.3)
        model = MockFeatureBasedBiasModel()
        df = create_test_data(20, 2)

        evaluation_result = self.create_mock_evaluation_result()
        evaluation_result.model_metadata["existing_key"] = "existing_value"

        processed_result = processor.process_results(model, df, "gt_idx", evaluation_result)

        # Verify existing metadata was preserved
        assert processed_result.model_metadata["existing_key"] == "existing_value"
        # And new metadata was added
        assert "zero_shot_baseline" in processed_result.model_metadata


class TestEvaluatorIntegration:
    """Integration tests for evaluators and post-processors"""

    def test_rf_pipeline_integration(self):
        """Test complete RF evaluation pipeline"""
        # Create components
        evaluator = RandomForestFoldEvaluator()
        post_processor = RandomForestPostProcessor()
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(40, 2)
        test_df = create_test_data(10, 2)

        # Run fold evaluation
        fold_result = evaluator.evaluate_fold(
            model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=1, seed=42
        )

        # Create evaluation result for post-processing
        from TsT.core.results import RepeatResult

        repeat_result = RepeatResult.from_fold_results(0, [fold_result])
        evaluation_result = EvaluationResult.from_repeat_results(
            model_name="integration_test", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

        # Run post-processing (with mocked sklearn components)
        with patch("TsT.evaluation._make_estimator") as mock_make_estimator:
            mock_estimator = Mock()
            mock_estimator.feature_importances_ = np.array([0.7, 0.3])
            mock_make_estimator.return_value = mock_estimator

            processed_result = post_processor.process_results(
                model=model, df=train_df, target_col="gt_idx", evaluation_result=evaluation_result
            )

        # Verify complete pipeline
        assert isinstance(processed_result, EvaluationResult)
        assert processed_result.feature_importances is not None
        assert len(processed_result.feature_importances) == 2
        assert "n_features" in processed_result.model_metadata

        # Verify fold result structure
        assert len(processed_result.repeat_results) == 1
        assert len(processed_result.repeat_results[0].fold_results) == 1
        assert processed_result.repeat_results[0].fold_results[0].fold_id == 1

    def test_llm_pipeline_integration(self):
        """Test complete LLM evaluation pipeline"""
        # Create components
        llm_config = {"model_name": "test/model", "batch_size": 4}
        evaluator = LLMFoldEvaluator(llm_config)
        post_processor = LLMPostProcessor(llm_config, zero_shot_baseline=0.25)
        model = MockFeatureBasedBiasModel()

        train_df = create_test_data(20, 2)
        test_df = create_test_data(5, 2)

        # Mock the legacy LLM evaluation function
        with patch("TsT.evaluation.evaluate_bias_model_llm") as mock_evaluate_llm:
            # Setup mock to return (mean_score, std_score, feature_importances, count)
            mock_evaluate_llm.return_value = (0.8, 0.1, None, 20)

            # Run fold evaluation
            fold_result = evaluator.evaluate_fold(
                model=model, train_df=train_df, test_df=test_df, target_col="gt_idx", fold_id=2, seed=123
            )

        # Create evaluation result
        from TsT.core.results import RepeatResult

        repeat_result = RepeatResult.from_fold_results(0, [fold_result])
        evaluation_result = EvaluationResult.from_repeat_results(
            model_name="llm_integration_test", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

        # Run post-processing
        processed_result = post_processor.process_results(
            model=model, df=train_df, target_col="gt_idx", evaluation_result=evaluation_result
        )

        # Verify complete pipeline
        assert isinstance(processed_result, EvaluationResult)
        assert processed_result.feature_importances is not None
        assert len(processed_result.feature_importances) == 3
        assert "zero_shot_baseline" in processed_result.model_metadata
        assert "improvement" in processed_result.model_metadata

        # Verify improvement calculation
        expected_improvement = 0.8 - 0.25  # LLM score - baseline
        assert processed_result.model_metadata["improvement"] == expected_improvement
