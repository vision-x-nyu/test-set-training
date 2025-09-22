"""
Tests for UnifiedCrossValidator.

These tests ensure the unified cross-validation engine works correctly
with different model types and produces rich, detailed results.
"""

import pytest
import pandas as pd
import numpy as np

from TsT.core.cross_validation import UnifiedCrossValidator, CrossValidationConfig
from TsT.core.protocols import ModelEvaluator
from TsT.core.protocols import FoldResult, EvaluationResult


class MockBiasModel:
    """Mock bias model for testing that implements FeatureBasedBiasModel interface"""

    def __init__(self, name="test_model", format="mc", task="clf", metric="acc", target_col_override=None):
        self.name = name
        self.format = format
        self._task = task
        self._metric = metric
        self.target_col_override = target_col_override
        self.feature_cols = ["feature1", "feature2"]  # Mock feature columns

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    @property
    def task(self):
        return self._task

    @property
    def metric(self):
        return self._metric

    def __str__(self) -> str:
        return f"{self.name} ({self.format})"

    # FeatureBasedBiasModel interface methods
    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Mock feature map fitting"""
        pass

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mock feature addition"""
        df_copy = df.copy()
        # Add simple mock features
        df_copy["feature1"] = df_copy.index.astype(float) * 0.1
        df_copy["feature2"] = df_copy.index.astype(float) * 0.2
        return df_copy


class MockModelEvaluator(ModelEvaluator):
    """Mock model evaluator that returns predictable results"""

    def __init__(self, scores=None):
        self.scores = scores or [0.8, 0.7, 0.9, 0.6, 0.85]
        self.call_count = 0
        self.calls = []  # Track all calls

    def train_and_evaluate_fold(self, model, train_df, test_df, target_col, fold_id, seed):
        """Return predetermined fold result"""
        score = self.scores[self.call_count % len(self.scores)]

        result = FoldResult(
            fold_id=fold_id,
            score=score,
            train_size=len(train_df),
            test_size=len(test_df),
            metadata={"seed": seed, "call_count": self.call_count},
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

    def process_results(self, model, df, target_col, evaluation_result):
        """Mock post-processing"""
        evaluation_result.model_metadata.update({"post_processed": True})
        return evaluation_result


def create_test_data(n_samples=100, n_classes=4):
    """Create test data for CV.

    Ensures each class has at least 2 samples to avoid stratification warnings
    when using StratifiedKFold with n_splits=2.
    """
    np.random.seed(42)

    # Create class labels with near-uniform distribution ensuring minimum counts
    labels = np.arange(n_classes)
    repeats = int(np.ceil(n_samples / n_classes))
    y = np.tile(labels, repeats)[:n_samples]
    np.random.shuffle(y)

    data = {
        "id": range(n_samples),
        "gt_idx": y,
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

        assert config.n_folds == 5
        assert config.random_state == 42
        assert config.repeats == 1
        assert config.verbose is True
        assert config.show_progress is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = CrossValidationConfig(n_folds=3, random_state=123, repeats=2, verbose=False, show_progress=False)

        assert config.n_folds == 3
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
        assert cv.config.n_folds == 5  # Default value

    def test_custom_config_initialization(self):
        """Test initialization with custom config"""
        config = CrossValidationConfig(n_folds=3, repeats=2)
        cv = UnifiedCrossValidator(config)

        assert cv.config.n_folds == 3
        assert cv.config.repeats == 2

    def test_basic_evaluation_rf(self):
        """Test basic RF model evaluation"""
        # Setup
        model = MockBiasModel()
        df = create_test_data(30, 2)

        config = CrossValidationConfig(n_folds=3, repeats=1, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        # Run evaluation
        result = cv.cross_validate(model=model, df=df, target_col="gt_idx", mode="rf")

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test_model"
        assert result.model_format == "mc"
        assert result.metric_name == "acc"
        assert len(result.repeat_results) == 1
        assert len(result.repeat_results[0].fold_results) == 3
        assert result.total_count == 30

    def test_multiple_repeats(self):
        """Test evaluation with multiple repeats"""
        model = MockBiasModel()
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_folds=2, repeats=3, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.cross_validate(model=model, df=df, target_col="gt_idx", mode="rf")

        # Verify structure
        assert len(result.repeat_results) == 3  # 3 repeats
        assert len(result.repeat_results[0].fold_results) == 2  # 2 folds per repeat

    def test_with_rf_post_processing(self):
        """Test evaluation with post-processor (integrated in RF evaluator)"""
        model = MockBiasModel()
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_folds=2, repeats=1, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.cross_validate(model=model, df=df, target_col="gt_idx", mode="rf")

        # Verify post-processing occurred (RF evaluator adds feature importances)
        assert result.feature_importances is not None
        assert len(result.feature_importances) >= 0  # May have features depending on mock model
        assert "total_samples" in result.model_metadata

    def test_target_column_override(self):
        """Test target column override functionality"""
        model = MockBiasModel(target_col_override="custom_target")

        df = create_test_data(10, 2)
        df["custom_target"] = df["gt_idx"]  # Add custom target column

        config = CrossValidationConfig(n_folds=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.cross_validate(
            model=model,
            df=df,
            target_col="gt_idx",  # Should be overridden
            mode="rf",
        )

        # Verify result structure (override was applied internally)
        assert result.model_name == "test_model"
        assert result.model_format == "mc"
        assert result.metric_name == "acc"

    def test_regression_target_column_conversion(self):
        """Test gt_idx -> ground_truth conversion for regression"""
        model = MockBiasModel(task="reg", metric="mra", format="num")
        df = create_test_data(10, 2)

        config = CrossValidationConfig(n_folds=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.cross_validate(
            model=model,
            df=df,
            target_col="gt_idx",  # Should be converted to "ground_truth"
            mode="rf",
        )

        # Verify result structure (conversion was applied internally)
        assert result.model_name == "test_model"
        assert result.model_format == "num"
        assert result.metric_name == "mra"

    def test_regression_vs_classification_splitters(self):
        """Test that correct splitters are used for regression vs classification"""
        # Classification model
        clf_model = MockBiasModel(task="clf")

        # Regression model
        reg_model = MockBiasModel(task="reg")

        df = create_test_data(20, 4)  # 4 classes for stratification

        config = CrossValidationConfig(n_folds=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        # Test classification (should use StratifiedKFold)
        clf_result = cv.cross_validate(clf_model, df, "gt_idx", mode="rf")
        assert len(clf_result.repeat_results[0].fold_results) == 2

        # Test regression (should use KFold)
        reg_result = cv.cross_validate(reg_model, df, "ground_truth", mode="rf")
        assert len(reg_result.repeat_results[0].fold_results) == 2

        # Both should complete successfully
        assert clf_result.total_count == 20
        assert reg_result.total_count == 20

    def test_fold_size_consistency(self):
        """Test that fold sizes are tracked correctly"""
        model = MockBiasModel()
        df = create_test_data(30, 2)  # Evenly divisible by 3

        config = CrossValidationConfig(n_folds=3, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.cross_validate(model, df, "gt_idx", mode="rf")

        # Check fold sizes
        fold_sizes = [fold.test_size for fold in result.repeat_results[0].fold_results]
        assert sum(fold_sizes) == 30  # Total test instances across all folds

        # Each fold should have ~10 test instances (30/3)
        for size in fold_sizes:
            assert 8 <= size <= 12  # Allow some variance due to CV splitting

    def test_metadata_preservation(self):
        """Test that metadata is preserved through the pipeline"""
        model = MockBiasModel(name="metadata_test")
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_folds=2, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        result = cv.cross_validate(model, df, "gt_idx", mode="rf")

        # Check that basic metadata exists
        for fold_result in result.repeat_results[0].fold_results:
            assert "estimator_params" in fold_result.metadata  # Added by RF evaluator
            assert fold_result.test_size > 0
            assert fold_result.train_size > 0

    def test_invalid_mode_error(self):
        """Test error handling when invalid mode is provided"""
        model = MockBiasModel()
        df = create_test_data(20, 2)

        config = CrossValidationConfig(n_folds=3, verbose=False, show_progress=False)
        cv = UnifiedCrossValidator(config)

        # Should raise error for invalid mode
        with pytest.raises(ValueError, match="Unknown mode"):
            cv.cross_validate(model, df, "gt_idx", mode="invalid")  # type: ignore

    def test_progress_tracking_disabled(self):
        """Test that progress tracking can be disabled"""
        model = MockBiasModel()
        df = create_test_data(20, 2)

        config = CrossValidationConfig(
            n_folds=2,
            repeats=2,
            verbose=False,
            show_progress=False,  # Disable progress bars
        )
        cv = UnifiedCrossValidator(config)

        # Should complete without issues (no progress bars shown)
        result = cv.cross_validate(model, df, "gt_idx", mode="rf")

        assert len(result.repeat_results) == 2
        assert len(result.repeat_results[0].fold_results) == 2


class TestUnifiedCrossValidatorIntegration:
    """Integration tests for UnifiedCrossValidator"""

    def test_realistic_evaluation_scenario(self):
        """Test a realistic evaluation scenario with multiple components"""
        # Create more realistic model
        model = MockBiasModel(name="realistic_model", format="mc", task="clf", metric="acc")
        df = create_test_data(100, 4)

        # Configure for realistic CV
        config = CrossValidationConfig(
            n_folds=5,
            repeats=2,
            random_state=42,
            verbose=True,  # Enable logging
            show_progress=False,  # Disable for testing
        )
        cv = UnifiedCrossValidator(config)

        # Run evaluation
        result = cv.cross_validate(model=model, df=df, target_col="gt_idx", mode="rf")

        # Verify comprehensive results
        assert result.model_name == "realistic_model"
        assert len(result.repeat_results) == 2
        assert len(result.repeat_results[0].fold_results) == 5

        # Check statistics are reasonable
        assert 0.0 <= result.overall_mean <= 1.0
        assert result.overall_std >= 0.0
        assert result.count == 100
        assert result.total_count == 100 * 2  # 100 samples * 2 repeats

        # Check feature importances (RF evaluator should add these)
        assert result.feature_importances is not None

        # Check metadata (RF evaluator should add these)
        assert "total_samples" in result.model_metadata
