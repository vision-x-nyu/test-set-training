"""
Tests for UnifiedCrossValidator.

These tests ensure the unified cross-validation engine works correctly
with different model types and produces rich, detailed results.
"""

import pytest
import pandas as pd
import numpy as np

from TsT.core.cross_validation import UnifiedCrossValidator, CrossValidationConfig, FoldEvaluator, PostProcessor
from TsT.core.results import FoldResult, EvaluationResult


class MockBiasModel:
    """Mock bias model for testing"""

    def __init__(self, name="test_model", format="mc", task="clf", metric="acc", target_col_override=None):
        self.name = name
        self.format = format
        self._task = task
        self._metric = metric
        self.target_col_override = target_col_override

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    @property
    def task(self):
        return self._task

    @property
    def metric(self):
        return self._metric


class MockFoldEvaluator(FoldEvaluator):
    """Mock fold evaluator that returns predictable results"""

    def __init__(self, scores=None):
        self.scores = scores or [0.8, 0.7, 0.9, 0.6, 0.85]
        self.call_count = 0
        self.calls = []  # Track all calls

    def evaluate_fold(self, model, train_df, test_df, target_col, fold_id, seed):
        """Return predetermined fold result"""
        score = self.scores[self.call_count % len(self.scores)]

        result = FoldResult(
            fold_id=fold_id,
            score=score,
            fold_size=len(test_df),
            metadata={"train_size": len(train_df), "seed": seed, "call_count": self.call_count},
        )

        self.calls.append(
            {
                "model": model,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "target_col": target_col,
                "fold_id": fold_id,
                "seed": seed,
            }
        )

        self.call_count += 1
        return result


class MockPostProcessor(PostProcessor):
    """Mock post-processor for testing"""

    def __init__(self, add_importances=True):
        self.add_importances = add_importances
        self.process_calls = []

    def process_results(self, model, df, target_col, evaluation_result):
        """Add mock processing to results"""
        self.process_calls.append({"model": model, "df_size": len(df), "target_col": target_col})

        if self.add_importances:
            # Add mock feature importances
            evaluation_result.feature_importances = pd.DataFrame(
                {"feature": ["feature_1", "feature_2"], "importance": [0.6, 0.4]}
            )

        # Add metadata
        evaluation_result.model_metadata.update({"post_processed": True, "processor_calls": len(self.process_calls)})

        return evaluation_result


def create_test_data(n_samples=100, n_classes=4):
    """Create test data for CV"""
    np.random.seed(42)

    data = {
        "id": range(n_samples),
        "gt_idx": np.random.randint(0, n_classes, n_samples),
        "ground_truth": np.random.randn(n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
    }

    return pd.DataFrame(data)


class TestCrossValidationConfig:
    """Test CrossValidationConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CrossValidationConfig()

        assert config.n_splits == 5
        assert config.random_state == 42
        assert config.repeats == 1
        assert config.verbose is True
        assert config.show_progress is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = CrossValidationConfig(n_splits=3, random_state=123, repeats=2, verbose=False, show_progress=False)

        assert config.n_splits == 3
        assert config.random_state == 123
        assert config.repeats == 2
        assert config.verbose is False
        assert config.show_progress is False


class TestUnifiedCrossValidator:
    """Test UnifiedCrossValidator functionality"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        cv = UnifiedCrossValidator()

        assert isinstance(cv.config, CrossValidationConfig)
        assert cv.config.n_splits == 5  # Default value

    def test_custom_config_initialization(self):
        """Test initialization with custom config"""
        config = CrossValidationConfig(n_splits=3, repeats=2)
        cv = UnifiedCrossValidator(config)

        assert cv.config.n_splits == 3
        assert cv.config.repeats == 2

    def test_basic_evaluation(self):
        """Test basic model evaluation"""
        # Setup
        model = MockBiasModel()
        evaluator = MockFoldEvaluator([0.8, 0.9, 0.7])
        df = create_test_data(30, 2)

        config = CrossValidationConfig(n_splits=3, repeats=1, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        # Run evaluation
        result = cv.evaluate_model(model=model, evaluator=evaluator, df=df, target_col="gt_idx")

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test_model"
        assert result.model_format == "mc"
        assert result.metric_name == "acc"
        assert len(result.repeat_results) == 1
        assert len(result.repeat_results[0].fold_results) == 3
        assert result.total_count == 30

        # Verify evaluator was called correctly
        assert evaluator.call_count == 3
        assert len(evaluator.calls) == 3

    def test_multiple_repeats(self):
        """Test evaluation with multiple repeats"""
        model = MockBiasModel()
        evaluator = MockFoldEvaluator([0.8, 0.9])  # 2 fold scores
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_splits=2, repeats=3, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.evaluate_model(model=model, evaluator=evaluator, df=df, target_col="gt_idx")

        # Verify structure
        assert len(result.repeat_results) == 3  # 3 repeats
        assert len(result.repeat_results[0].fold_results) == 2  # 2 folds per repeat
        assert evaluator.call_count == 6  # 3 repeats × 2 folds

        # Verify different seeds for different repeats
        seeds = [call["seed"] for call in evaluator.calls]
        unique_seeds = set(seeds)
        assert len(unique_seeds) == 3  # Should have 3 different seeds (one per repeat)

    def test_with_post_processor(self):
        """Test evaluation with post-processor"""
        model = MockBiasModel()
        evaluator = MockFoldEvaluator([0.8, 0.7])
        post_processor = MockPostProcessor()
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_splits=2, repeats=1, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.evaluate_model(
            model=model, evaluator=evaluator, df=df, target_col="gt_idx", post_processor=post_processor
        )

        # Verify post-processing occurred
        assert len(post_processor.process_calls) == 1
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 2
        assert result.model_metadata["post_processed"] is True

    def test_target_column_override(self):
        """Test target column override functionality"""
        model = MockBiasModel(target_col_override="custom_target")
        evaluator = MockFoldEvaluator([0.8])

        df = create_test_data(10, 2)
        df["custom_target"] = df["gt_idx"]  # Add custom target column

        config = CrossValidationConfig(n_splits=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.evaluate_model(
            model=model,
            evaluator=evaluator,
            df=df,
            target_col="gt_idx",  # Should be overridden
        )

        # Verify override was applied
        assert all(call["target_col"] == "custom_target" for call in evaluator.calls)
        assert result.model_name == "test_model"
        assert result.model_format == "mc"
        assert result.metric_name == "acc"

    def test_regression_target_column_conversion(self):
        """Test gt_idx -> ground_truth conversion for regression"""
        model = MockBiasModel(task="reg", metric="mra", format="num")
        evaluator = MockFoldEvaluator([0.7])
        df = create_test_data(10, 2)

        config = CrossValidationConfig(n_splits=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.evaluate_model(
            model=model,
            evaluator=evaluator,
            df=df,
            target_col="gt_idx",  # Should be converted to "ground_truth"
        )

        # Verify conversion was applied
        assert all(call["target_col"] == "ground_truth" for call in evaluator.calls)
        assert result.model_name == "test_model"
        assert result.model_format == "num"
        assert result.metric_name == "mra"

    def test_regression_vs_classification_splitters(self):
        """Test that correct splitters are used for regression vs classification"""
        # Classification model
        clf_model = MockBiasModel(task="clf")
        clf_evaluator = MockFoldEvaluator([0.8])

        # Regression model
        reg_model = MockBiasModel(task="reg")
        reg_evaluator = MockFoldEvaluator([0.7])

        df = create_test_data(20, 4)  # 4 classes for stratification

        config = CrossValidationConfig(n_splits=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        # Test classification (should use StratifiedKFold)
        clf_result = cv.evaluate_model(clf_model, clf_evaluator, df, "gt_idx")
        assert len(clf_result.repeat_results[0].fold_results) == 2

        # Test regression (should use KFold)
        reg_result = cv.evaluate_model(reg_model, reg_evaluator, df, "ground_truth")
        assert len(reg_result.repeat_results[0].fold_results) == 2

        # Both should complete successfully
        assert clf_result.total_count == 20
        assert reg_result.total_count == 20

    def test_fold_size_consistency(self):
        """Test that fold sizes are tracked correctly"""
        model = MockBiasModel()
        evaluator = MockFoldEvaluator([0.8, 0.9, 0.7])
        df = create_test_data(30, 2)  # Evenly divisible by 3

        config = CrossValidationConfig(n_splits=3, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.evaluate_model(model, evaluator, df, "gt_idx")

        # Check fold sizes
        fold_sizes = [fold.fold_size for fold in result.repeat_results[0].fold_results]
        assert sum(fold_sizes) == 30  # Total test instances across all folds

        # Each fold should have ~10 test instances (30/3)
        for size in fold_sizes:
            assert 8 <= size <= 12  # Allow some variance due to CV splitting

    def test_metadata_preservation(self):
        """Test that metadata is preserved through the pipeline"""
        model = MockBiasModel(name="metadata_test")

        # Create evaluator that adds metadata
        class MetadataEvaluator(MockFoldEvaluator):
            def evaluate_fold(self, model, train_df, test_df, target_col, fold_id, seed):
                result = super().evaluate_fold(model, train_df, test_df, target_col, fold_id, seed)
                result.metadata.update({"model_name": model.name, "evaluation_timestamp": "2024-test"})
                return result

        evaluator = MetadataEvaluator([0.8, 0.9])
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_splits=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.evaluate_model(model, evaluator, df, "gt_idx")

        # Check that metadata was preserved
        for fold_result in result.repeat_results[0].fold_results:
            assert fold_result.metadata["model_name"] == "metadata_test"
            assert fold_result.metadata["evaluation_timestamp"] == "2024-test"

    def test_error_handling_in_evaluator(self):
        """Test error handling when evaluator fails"""
        model = MockBiasModel()

        # Create evaluator that fails on second call
        class FailingEvaluator(FoldEvaluator):
            def __init__(self):
                self.call_count = 0

            def evaluate_fold(self, model, train_df, test_df, target_col, fold_id, seed):
                self.call_count += 1
                if self.call_count == 2:
                    raise ValueError("Simulated evaluator failure")

                return FoldResult(fold_id, 0.8, len(test_df))

        evaluator = FailingEvaluator()
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_splits=3, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        # Should propagate the error
        with pytest.raises(ValueError, match="Simulated evaluator failure"):
            cv.evaluate_model(model, evaluator, df, "gt_idx")

    def test_progress_tracking_disabled(self):
        """Test that progress tracking can be disabled"""
        model = MockBiasModel()
        evaluator = MockFoldEvaluator([0.8, 0.9])
        df = create_test_data(20, 2)

        config = CrossValidationConfig(
            n_splits=2,
            repeats=2,
            verbose=False,
            show_progress=False,  # Disable progress bars
        )
        cv = UnifiedCrossValidator(config)

        # Should complete without issues (no progress bars shown)
        result = cv.evaluate_model(model, evaluator, df, "gt_idx")

        assert len(result.repeat_results) == 2
        assert len(result.repeat_results[0].fold_results) == 2


class TestUnifiedCrossValidatorIntegration:
    """Integration tests for UnifiedCrossValidator"""

    def test_realistic_evaluation_scenario(self):
        """Test a realistic evaluation scenario with multiple components"""
        # Create more realistic model
        model = MockBiasModel(name="realistic_model", format="mc", task="clf", metric="acc")

        # Create evaluator with realistic score distribution
        realistic_scores = [0.75, 0.82, 0.68, 0.79, 0.84]  # 5 folds
        evaluator = MockFoldEvaluator(realistic_scores)

        # Create post-processor that adds comprehensive metadata
        class RealisticPostProcessor(PostProcessor):
            def process_results(self, model, df, target_col, evaluation_result):
                # Add realistic feature importances
                evaluation_result.feature_importances = pd.DataFrame(
                    {"feature": [f"feature_{i}" for i in range(5)], "importance": [0.3, 0.25, 0.2, 0.15, 0.1]}
                )

                # Add comprehensive metadata
                evaluation_result.model_metadata.update(
                    {
                        "n_features": 5,
                        "total_samples": len(df),
                        "model_type": "mock_classifier",
                        "evaluation_version": "1.0",
                    }
                )

                return evaluation_result

        post_processor = RealisticPostProcessor()
        df = create_test_data(100, 4)

        # Configure for realistic CV
        config = CrossValidationConfig(
            n_splits=5,
            repeats=2,
            random_state=42,
            verbose=True,  # Enable logging
            show_progress=False,  # Disable for testing
        )
        cv = UnifiedCrossValidator(config)

        # Run evaluation
        result = cv.evaluate_model(
            model=model, evaluator=evaluator, df=df, target_col="gt_idx", post_processor=post_processor
        )

        # Verify comprehensive results
        assert result.model_name == "realistic_model"
        assert len(result.repeat_results) == 2
        assert len(result.repeat_results[0].fold_results) == 5

        # Check statistics are reasonable
        assert 0.6 <= result.overall_mean <= 0.9
        assert result.overall_std >= 0.0
        assert result.total_count == 100

        # Check feature importances
        assert len(result.feature_importances) == 5
        assert abs(result.feature_importances["importance"].sum() - 1.0) < 1e-10

        # Check metadata
        assert result.model_metadata["n_features"] == 5
        assert result.model_metadata["total_samples"] == 100

        # Check evaluator was called correctly
        assert evaluator.call_count == 10  # 2 repeats × 5 folds

    def test_comparison_with_legacy_approach(self):
        """Test that unified approach produces consistent results"""
        # This is a conceptual test - in practice you'd compare with actual legacy results

        model = MockBiasModel()
        evaluator = MockFoldEvaluator([0.8, 0.85, 0.75, 0.9, 0.7])
        df = create_test_data(50, 2)

        config = CrossValidationConfig(n_splits=5, repeats=1, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.evaluate_model(model, evaluator, df, "gt_idx")

        # Convert to legacy format for comparison
        legacy_tuple = result.to_legacy_tuple()

        # Verify legacy compatibility
        assert len(legacy_tuple) == 4
        assert legacy_tuple[0] == result.overall_mean  # mean_score
        assert legacy_tuple[1] == result.overall_std  # std_score
        assert legacy_tuple[3] == result.total_count  # count

        # Check that means match expected calculation
        expected_mean = np.mean([0.8, 0.85, 0.75, 0.9, 0.7])
        assert abs(result.overall_mean - expected_mean) < 1e-10
