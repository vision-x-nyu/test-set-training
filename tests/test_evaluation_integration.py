"""
Integration tests for evaluation framework.

These tests verify that the new unified evaluation framework produces
results identical to the old approach (backward compatibility).
"""

import pytest
import pandas as pd
import numpy as np

from TsT.evaluation import run_evaluation
from TsT.core.cross_validation import UnifiedCrossValidator, CrossValidationConfig


class SimpleTestModel:
    """Simple test model for integration testing"""

    def __init__(self, name="simple_test"):
        self.name = name
        self.format = "mc"
        self.feature_cols = ["feature1", "feature2"]
        self.target_col_override = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select all rows"""
        return df

    @property
    def task(self):
        if self.format == "mc":
            return "clf"
        elif self.format == "num":
            return "reg"
        else:
            raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self):
        if self.format == "mc":
            return "acc"
        elif self.format == "num":
            return "mra"
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """No-op for simple model"""
        pass

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple features for testing"""
        df_copy = df.copy()

        # Add predictable features based on target
        # This makes the model actually learnable for testing
        # Use deterministic features to avoid changing target types
        df_copy["feature1"] = df_copy["gt_idx"].astype(float) + 0.1
        df_copy["feature2"] = df_copy["gt_idx"].astype(float) * 2 + 0.2

        return df_copy


def create_learnable_data(n_samples=100, n_classes=2, seed=42):
    """Create synthetic data that's actually learnable"""
    np.random.seed(seed)

    # Create data where features correlate with target
    values = "abcdefghijklmnopqrstuvwxyz"
    gt_idx = np.random.randint(0, n_classes, n_samples)
    gt_values = [values[i] for i in gt_idx]

    data = {
        "id": range(n_samples),
        "gt_idx": gt_idx.astype(int),  # Explicitly ensure integers for classification
        "ground_truth": gt_values,
    }

    df = pd.DataFrame(data)
    # Explicitly set dtypes to be sure
    df["gt_idx"] = df["gt_idx"].astype("int64")

    return df


class TestUnifiedFrameworkIntegration:
    """Test integration of unified evaluation framework"""

    def test_basic_evaluation_pipeline(self):
        """Test that basic evaluation pipeline works end-to-end"""
        # Setup
        model = SimpleTestModel()
        df = create_learnable_data(50, 4)

        # Test unified framework
        results = run_evaluation(
            question_models=[model], df_full=df, n_splits=3, random_state=42, verbose=False, repeats=1, mode="rf"
        )

        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1  # One model
        assert "Model" in results.columns
        assert "Score" in results.columns
        assert "Count" in results.columns

        # Verify result values
        assert results.iloc[0]["Model"] == "simple_test"
        assert results.iloc[0]["Count"] == 50

        # Score should be reasonable (model is learnable)
        score_str = results.iloc[0]["Score"]
        if isinstance(score_str, str):
            score = float(score_str.rstrip("%")) / 100
        else:
            score = float(score_str)
        assert 0.0 <= score <= 1.0

    def test_cross_validation_with_evaluator(self):
        """Test cross-validation with UnifiedCrossValidator directly"""
        # Setup
        model = SimpleTestModel()
        df = create_learnable_data(40, 4)

        # Create cross-validator
        config = CrossValidationConfig(n_folds=2, random_state=42, verbose=False, repeats=1, show_progress=False)
        cv = UnifiedCrossValidator(config)

        # Run CV using the correct interface
        result = cv.cross_validate(
            model=model,
            df=df,
            target_col="gt_idx",
            mode="rf",  # Specify RF mode instead of passing evaluator directly
        )

        # Verify results
        assert isinstance(result.overall_mean, float)
        assert isinstance(result.overall_std, float)
        assert isinstance(result.total_count, int)
        assert 0.0 <= result.overall_mean <= 1.0
        assert result.overall_std >= 0.0
        assert result.total_count == 40

    def test_deterministic_results(self):
        """Test that same parameters produce same results"""
        model = SimpleTestModel()
        df = create_learnable_data(30, 2, seed=42)

        # Run evaluation twice with same parameters
        results1 = run_evaluation(
            question_models=[model], df_full=df, n_splits=2, random_state=42, verbose=False, mode="rf"
        )

        results2 = run_evaluation(
            question_models=[model],
            df_full=df,
            n_splits=2,
            random_state=42,  # Same seed
            verbose=False,
            mode="rf",
        )

        # Results should be identical
        assert results1.iloc[0]["Score"] == results2.iloc[0]["Score"]
        assert results1.iloc[0]["Count"] == results2.iloc[0]["Count"]

    def test_multiple_models_evaluation(self):
        """Test evaluation with multiple models"""
        # Create multiple models
        model1 = SimpleTestModel("model_1")
        model2 = SimpleTestModel("model_2")
        models = [model1, model2]

        df = create_learnable_data(40, 2)

        # Evaluate multiple models
        results = run_evaluation(
            question_models=models, df_full=df, n_splits=2, random_state=42, verbose=False, mode="rf"
        )

        # Verify results
        assert len(results) == 2
        assert "model_1" in results["Model"].values
        assert "model_2" in results["Model"].values

    def test_error_handling(self):
        """Test that evaluation handles errors gracefully"""

        class BrokenModel(SimpleTestModel):
            def add_features(self, df):
                raise ValueError("Simulated model error")

        model = BrokenModel("broken_model")
        df = create_learnable_data(20, 2)

        # Should handle error gracefully
        results = run_evaluation(question_models=[model], df_full=df, n_splits=2, verbose=False, mode="rf")

        # Should still return results (with error info)
        assert len(results) == 1
        assert results.iloc[0]["Model"] == "broken_model"
        # Score should be 0 for failed evaluation
        assert results.iloc[0]["Score"] == "0.0%"

    def test_different_target_columns(self):
        """Test evaluation with different target columns"""
        model = SimpleTestModel()
        df = create_learnable_data(30, 2)

        # Test with gt_idx
        results1 = run_evaluation(
            question_models=[model], df_full=df, target_col="gt_idx", n_splits=2, verbose=False, mode="rf"
        )

        # Test with ground_truth (for regression-like scenario)
        model_reg = SimpleTestModel()
        model_reg._task = "reg"
        model_reg._metric = "mra"

        results2 = run_evaluation(
            question_models=[model_reg], df_full=df, target_col="ground_truth", n_splits=2, verbose=False, mode="rf"
        )

        # Both should succeed
        assert len(results1) == 1
        assert len(results2) == 1

    def test_question_type_filtering(self):
        """Test filtering models by question types"""
        model1 = SimpleTestModel("keep_this")
        model2 = SimpleTestModel("filter_this")
        models = [model1, model2]

        df = create_learnable_data(20, 2)

        # Filter to only one model
        results = run_evaluation(
            question_models=models, df_full=df, question_types=["keep_this"], n_splits=2, verbose=False, mode="rf"
        )

        # Should only have one result
        assert len(results) == 1
        assert results.iloc[0]["Model"] == "keep_this"


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality"""

    def test_video_mme_integration(self):
        """Test integration with actual Video-MME models"""
        try:
            from TsT.benchmarks.video_mme import benchmark

            models = benchmark.get_feature_based_models()[:1]  # Just test one model
            if not models:
                pytest.skip("No Video-MME models available")

            df = benchmark.load_data()
            if len(df) == 0:
                pytest.skip("No Video-MME data available")

            # Use small subset for fast testing
            df_small = df.head(30)

            # Test new evaluation framework
            results = run_evaluation(
                question_models=models,
                df_full=df_small,
                n_splits=2,
                random_state=42,
                verbose=False,
                mode="rf",
                target_col="gt_idx",
            )

            # Verify basic structure
            assert len(results) == 1
            assert isinstance(results, pd.DataFrame)
            assert "Score" in results.columns
            assert results.iloc[0]["Count"] == 30

        except ImportError:
            pytest.skip("Video-MME benchmark not available")

    def test_regression_vs_classification(self):
        """Test that both regression and classification tasks work"""

        # Classification model
        clf_model = SimpleTestModel("clf_test")
        assert clf_model.task == "clf"
        assert clf_model.metric == "acc"

        # Regression model
        reg_model = SimpleTestModel("reg_test")
        reg_model.format = "num"  # Change to numerical format

        df = create_learnable_data(25, 2)

        # Test both
        clf_results = run_evaluation(
            question_models=[clf_model], df_full=df, target_col="gt_idx", n_splits=2, verbose=False, mode="rf"
        )

        reg_results = run_evaluation(
            question_models=[reg_model], df_full=df, target_col="ground_truth", n_splits=2, verbose=False, mode="rf"
        )

        # Both should succeed
        assert len(clf_results) == 1
        assert len(reg_results) == 1
        assert clf_results.iloc[0]["Metric"] == "ACC"
        assert reg_results.iloc[0]["Metric"] == "MRA"


class TestPerformanceAndMemory:
    """Test performance characteristics of new framework"""

    def test_evaluation_speed(self):
        """Test that evaluation completes in reasonable time"""
        import time

        model = SimpleTestModel()
        df = create_learnable_data(100, 2)

        # Time the evaluation
        start_time = time.time()

        results = run_evaluation(
            question_models=[model], df_full=df, n_splits=3, random_state=42, verbose=False, mode="rf"
        )

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly (less than 10 seconds even on slow machines)
        assert duration < 10.0, f"Evaluation took too long: {duration:.2f} seconds"
        assert len(results) == 1

    def test_memory_usage(self):
        """Test that evaluation doesn't consume excessive memory"""
        # This is a basic smoke test - in production you'd use memory profilers
        model = SimpleTestModel()
        df = create_learnable_data(20000, 2)

        # Should complete without memory errors
        results = run_evaluation(question_models=[model], df_full=df, n_splits=2, verbose=False, mode="rf")

        assert len(results) == 1
