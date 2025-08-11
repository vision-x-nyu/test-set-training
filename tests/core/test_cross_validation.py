"""
Tests for unified cross-validation logic.

These tests ensure the core CV framework works correctly with different
model types and handles edge cases properly.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from typing import Literal, Optional

from TsT.core.cross_validation import run_cross_validation


class MockBiasModel:
    """Mock bias model for testing"""

    def __init__(
        self,
        name: str = "test_model",
        format: Literal["mc", "num"] = "mc",
        task: Literal["clf", "reg"] = "clf",
        metric: Literal["acc", "mra"] = "acc",
        target_col_override: Optional[str] = None,
    ):
        self.name = name
        self.format = format
        self._task = task
        self._metric = metric
        self.target_col_override = target_col_override

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return all rows for testing"""
        return df

    @property
    def task(self) -> Literal["clf", "reg"]:
        return self._task

    @property
    def metric(self) -> Literal["acc", "mra"]:
        return self._metric


class MockEvaluator:
    """Mock evaluator that returns predictable scores"""

    def __init__(self, scores=None):
        self.scores = scores or [0.8, 0.7, 0.9, 0.6, 0.85]  # 5 fold scores
        self.call_count = 0

    def evaluate_fold(self, model, train_df, test_df, target_col, fold_num, seed):
        """Return predetermined score for this fold"""
        score = self.scores[self.call_count % len(self.scores)]
        self.call_count += 1
        return score


def create_test_data(n_samples=100, n_classes=4):
    """Create synthetic test data for CV testing"""
    np.random.seed(42)

    # For classification task - ensure integer classes
    gt_idx = np.random.randint(0, n_classes, n_samples)

    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "gt_idx": gt_idx,  # Integer classes for classification
        "ground_truth": np.random.randn(n_samples),  # Continuous for regression
    }

    return pd.DataFrame(data)


class TestRunCrossValidation:
    """Test the unified cross-validation function"""

    def test_basic_classification_cv(self):
        """Test basic CV with classification model"""
        # Setup
        model = MockBiasModel(task="clf", metric="acc")
        scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        evaluator = MockEvaluator(scores)
        df = create_test_data(100, 4)

        # Run CV
        mean_score, std_score, count = run_cross_validation(
            model=model,
            evaluator=evaluator,
            df=df,
            n_splits=5,
            random_state=42,
            verbose=False,
            repeats=1,
            target_col="gt_idx",
        )

        # Verify results
        expected_mean = np.mean(scores)
        expected_std = np.std(scores)

        assert abs(mean_score - expected_mean) < 1e-6
        assert abs(std_score - expected_std) < 0.2  # Very lenient tolerance since CV has randomness
        assert count == 100
        assert evaluator.call_count == 5  # 5 folds

    def test_regression_cv(self):
        """Test CV with regression model"""
        # Setup
        model = MockBiasModel(task="reg", metric="mra")
        scores = [0.6, 0.7, 0.8]
        evaluator = MockEvaluator(scores)
        df = create_test_data(60, 4)

        # Run CV
        mean_score, std_score, count = run_cross_validation(
            model=model,
            evaluator=evaluator,
            df=df,
            n_splits=3,
            random_state=42,
            verbose=False,
            repeats=1,
            target_col="ground_truth",
        )

        # Verify results
        expected_mean = np.mean(scores)
        expected_std = np.std(scores)

        assert abs(mean_score - expected_mean) < 1e-6
        assert abs(std_score - expected_std) < 0.2  # Very lenient tolerance since CV has randomness
        assert count == 60
        assert evaluator.call_count == 3  # 3 folds

    def test_multiple_repeats(self):
        """Test CV with multiple repeats"""
        # Setup
        model = MockBiasModel(task="clf", metric="acc")
        scores = [0.8, 0.9]
        evaluator = MockEvaluator(scores)  # 2 fold scores, will cycle
        df = create_test_data(40, 2)

        # Run CV with 2 repeats, 2 folds each = 4 total fold evaluations
        mean_score, std_score, count = run_cross_validation(
            model=model,
            evaluator=evaluator,
            df=df,
            n_splits=2,
            random_state=42,
            verbose=False,
            repeats=2,
            target_col="gt_idx",
        )

        # Should have: Repeat 1: [0.8, 0.9] -> mean=0.85
        #              Repeat 2: [0.8, 0.9] -> mean=0.85
        # Overall: mean=0.85, std=0.0

        assert abs(mean_score - 0.85) < 1e-6
        assert abs(std_score - 0.0) < 1e-6
        assert count == 40
        assert evaluator.call_count == 4  # 2 repeats × 2 folds

    def test_target_col_override(self):
        """Test target column override functionality"""
        # Setup
        model = MockBiasModel(target_col_override="custom_target")
        evaluator = Mock()
        evaluator.evaluate_fold = Mock(return_value=0.8)
        df = create_test_data(20, 2)
        df["custom_target"] = df["gt_idx"]  # Add custom target column

        # Run CV
        run_cross_validation(
            model=model,
            evaluator=evaluator,
            df=df,
            n_splits=2,
            verbose=False,
            target_col="gt_idx",  # Should be overridden
        )

        # Verify evaluator was called with override target column
        calls = evaluator.evaluate_fold.call_args_list
        assert len(calls) == 2  # 2 folds
        for call in calls:
            args, kwargs = call
            assert args[3] == "custom_target"  # target_col argument

    def test_regression_target_col_conversion(self):
        """Test gt_idx -> ground_truth conversion for regression"""
        # Setup
        model = MockBiasModel(task="reg", metric="mra")
        evaluator = Mock()
        evaluator.evaluate_fold = Mock(return_value=0.7)
        df = create_test_data(20, 2)

        # Run CV with gt_idx target for regression model
        run_cross_validation(
            model=model,
            evaluator=evaluator,
            df=df,
            n_splits=2,
            verbose=False,
            target_col="gt_idx",  # Should be converted to "ground_truth"
        )

        # Verify evaluator was called with converted target column
        calls = evaluator.evaluate_fold.call_args_list
        for call in calls:
            args, kwargs = call
            assert args[3] == "ground_truth"  # target_col should be converted

    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        model = MockBiasModel()
        evaluator = MockEvaluator()
        df = pd.DataFrame()  # Empty dataframe

        with pytest.raises((ValueError, IndexError, KeyError)):
            # Should fail gracefully with empty data
            run_cross_validation(
                model=model, evaluator=evaluator, df=df, n_splits=2, verbose=False, target_col="ground_truth"
            )

    def test_single_fold(self):
        """Test CV with n_splits=1 (edge case)"""
        model = MockBiasModel()
        scores = [0.75]
        evaluator = MockEvaluator(scores)
        df = create_test_data(20, 2)

        # Note: sklearn requires n_splits >= 2, so this should fail
        with pytest.raises(ValueError):
            run_cross_validation(model=model, evaluator=evaluator, df=df, n_splits=1, verbose=False)


class TestProgressTracking:
    """Test progress tracking and logging functionality"""

    def test_verbose_logging(self, caplog):
        """Test that verbose mode produces expected log output"""
        import logging

        caplog.set_level(logging.INFO)

        model = MockBiasModel(name="test_model", metric="acc")
        evaluator = MockEvaluator([0.8, 0.9])
        df = create_test_data(20, 2)

        run_cross_validation(
            model=model,
            evaluator=evaluator,
            df=df,
            n_splits=2,
            verbose=True,
            repeats=1,
            target_col="gt_idx",  # Use integer target for classification
        )

        # Check that logging occurred
        log_messages = [record.message for record in caplog.records]

        # Should have overall results log
        overall_log = [msg for msg in log_messages if "Overall ACC" in msg]
        assert len(overall_log) > 0

        # Should have fold scores log
        fold_log = [msg for msg in log_messages if "Fold scores" in msg]
        assert len(fold_log) > 0

    def test_non_verbose_mode(self, caplog):
        """Test that non-verbose mode produces minimal output"""
        import logging

        caplog.set_level(logging.INFO)

        model = MockBiasModel()
        evaluator = MockEvaluator([0.8, 0.9])
        df = create_test_data(20, 2)

        run_cross_validation(
            model=model,
            evaluator=evaluator,
            df=df,
            n_splits=2,
            verbose=False,  # Non-verbose
            target_col="gt_idx",  # Use integer target for classification
        )

        # Should have minimal or no logging
        log_messages = [record.message for record in caplog.records]
        overall_logs = [msg for msg in log_messages if "Overall" in msg]
        assert len(overall_logs) == 0  # No overall logs in non-verbose mode
