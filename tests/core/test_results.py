"""
Tests for unified evaluation result objects.

These tests ensure the result data structures work correctly and provide
expected functionality for aggregating and converting evaluation results.
"""

import pandas as pd
import numpy as np

from TsT.core.results import FoldResult, RepeatResult, EvaluationResult


class TestFoldResult:
    """Test FoldResult functionality"""

    def test_basic_creation(self):
        """Test basic FoldResult creation"""
        result = FoldResult(fold_id=1, score=0.85, fold_size=20, metadata={"test": True})

        assert result.fold_id == 1
        assert result.score == 0.85
        assert result.fold_size == 20
        assert result.metadata["test"] is True

    def test_default_metadata(self):
        """Test FoldResult with default metadata"""
        result = FoldResult(fold_id=2, score=0.7, fold_size=15)

        assert result.metadata == {}
        assert isinstance(result.metadata, dict)

    def test_metadata_modification(self):
        """Test that metadata can be modified after creation"""
        result = FoldResult(fold_id=1, score=0.8, fold_size=10)
        result.metadata["new_key"] = "new_value"

        assert result.metadata["new_key"] == "new_value"


class TestRepeatResult:
    """Test RepeatResult functionality"""

    def test_basic_creation(self):
        """Test basic RepeatResult creation"""
        fold_results = [
            FoldResult(1, 0.8, 20),
            FoldResult(2, 0.9, 20),
            FoldResult(3, 0.7, 20),
        ]

        result = RepeatResult(repeat_id=0, fold_results=fold_results, mean_score=0.8, std_score=0.1)

        assert result.repeat_id == 0
        assert len(result.fold_results) == 3
        assert result.mean_score == 0.8
        assert result.std_score == 0.1

    def test_total_instances(self):
        """Test total_instances property"""
        fold_results = [
            FoldResult(1, 0.8, 20),
            FoldResult(2, 0.9, 15),
            FoldResult(3, 0.7, 25),
        ]

        result = RepeatResult(0, fold_results, 0.8, 0.1)
        assert result.total_instances == 60  # 20 + 15 + 25

    def test_from_fold_results(self):
        """Test creating RepeatResult from fold results with calculated statistics"""
        fold_results = [
            FoldResult(1, 0.8, 20),
            FoldResult(2, 0.6, 20),
            FoldResult(3, 1.0, 20),
        ]

        result = RepeatResult.from_fold_results(0, fold_results)

        assert result.repeat_id == 0
        assert len(result.fold_results) == 3

        # Check calculated statistics
        expected_mean = (0.8 + 0.6 + 1.0) / 3
        expected_std = np.std([0.8, 0.6, 1.0])

        assert abs(result.mean_score - expected_mean) < 1e-6
        assert abs(result.std_score - expected_std) < 1e-6

    def test_empty_fold_results(self):
        """Test RepeatResult with empty fold results"""
        result = RepeatResult.from_fold_results(0, [])

        assert result.repeat_id == 0
        assert len(result.fold_results) == 0
        assert result.total_instances == 0
        # Mean and std of empty list should be 0
        assert result.mean_score == 0.0
        assert result.std_score == 0.0


class TestEvaluationResult:
    """Test EvaluationResult functionality"""

    def create_sample_repeat_results(self):
        """Create sample repeat results for testing"""
        # First repeat
        fold_results_1 = [
            FoldResult(1, 0.8, 25),
            FoldResult(2, 0.9, 25),
        ]
        repeat_1 = RepeatResult.from_fold_results(0, fold_results_1)

        # Second repeat
        fold_results_2 = [
            FoldResult(1, 0.7, 25),
            FoldResult(2, 0.8, 25),
        ]
        repeat_2 = RepeatResult.from_fold_results(1, fold_results_2)

        return [repeat_1, repeat_2]

    def test_basic_creation(self):
        """Test basic EvaluationResult creation"""
        repeat_results = self.create_sample_repeat_results()

        result = EvaluationResult(
            model_name="test_model",
            model_format="mc",
            metric_name="acc",
            repeat_results=repeat_results,
            overall_mean=0.825,
            overall_std=0.025,
            total_count=100,
        )

        assert result.model_name == "test_model"
        assert result.model_format == "mc"
        assert result.metric_name == "acc"
        assert len(result.repeat_results) == 2
        assert result.overall_mean == 0.825
        assert result.overall_std == 0.025
        assert result.total_count == 100

    def test_from_repeat_results(self):
        """Test creating EvaluationResult from repeat results with calculated statistics"""
        repeat_results = self.create_sample_repeat_results()

        result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Check calculated statistics
        # Repeat 1 mean: (0.8 + 0.9) / 2 = 0.85
        # Repeat 2 mean: (0.7 + 0.8) / 2 = 0.75
        # Overall mean: (0.85 + 0.75) / 2 = 0.8
        # Overall std: std([0.85, 0.75]) = 0.05

        assert abs(result.overall_mean - 0.8) < 1e-6
        assert abs(result.overall_std - 0.05) < 1e-6
        assert result.total_count == 50  # 25 * 2 folds

    def test_to_summary_dict(self):
        """Test conversion to summary dictionary"""
        repeat_results = self.create_sample_repeat_results()

        result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        summary = result.to_summary_dict()

        assert summary["Model"] == "test_model"
        assert summary["Format"] == "MC"
        assert summary["Metric"] == "ACC"
        assert summary["Score"] == "80.0%"  # 0.8 formatted as percentage
        assert summary["± Std"] == "5.0%"  # 0.05 formatted as percentage
        assert summary["Count"] == 50

    def test_to_legacy_tuple(self):
        """Test conversion to legacy tuple format"""
        repeat_results = self.create_sample_repeat_results()

        # Add some feature importances
        feature_importances = pd.DataFrame({"feature": ["feat1", "feat2"], "importance": [0.6, 0.4]})

        result = EvaluationResult.from_repeat_results(
            model_name="test_model",
            model_format="mc",
            metric_name="acc",
            repeat_results=repeat_results,
            feature_importances=feature_importances,
        )

        legacy_tuple = result.to_legacy_tuple()

        assert len(legacy_tuple) == 4
        assert abs(legacy_tuple[0] - 0.8) < 1e-6  # mean_score
        assert abs(legacy_tuple[1] - 0.05) < 1e-6  # std_score
        assert isinstance(legacy_tuple[2], pd.DataFrame)  # feature_importances
        assert legacy_tuple[3] == 50  # count

    def test_with_metadata(self):
        """Test EvaluationResult with model metadata"""
        repeat_results = self.create_sample_repeat_results()

        metadata = {"n_features": 10, "model_type": "random_forest"}

        result = EvaluationResult.from_repeat_results(
            model_name="test_model",
            model_format="mc",
            metric_name="acc",
            repeat_results=repeat_results,
            model_metadata=metadata,
        )

        assert result.model_metadata["n_features"] == 10
        assert result.model_metadata["model_type"] == "random_forest"

    def test_empty_repeat_results(self):
        """Test EvaluationResult with empty repeat results"""
        result = EvaluationResult.from_repeat_results(
            model_name="empty_model", model_format="mc", metric_name="acc", repeat_results=[]
        )

        assert result.overall_mean == 0.0
        assert result.overall_std == 0.0
        assert result.total_count == 0


class TestResultsIntegration:
    """Test integration between different result types"""

    def test_complete_result_pipeline(self):
        """Test complete pipeline from fold results to evaluation result"""
        # Create fold results for two repeats
        fold_results_1 = [
            FoldResult(1, 0.9, 30, {"estimator": "rf_1"}),
            FoldResult(2, 0.8, 30, {"estimator": "rf_2"}),
            FoldResult(3, 0.85, 30, {"estimator": "rf_3"}),
        ]

        fold_results_2 = [
            FoldResult(1, 0.95, 30, {"estimator": "rf_1"}),
            FoldResult(2, 0.75, 30, {"estimator": "rf_2"}),
            FoldResult(3, 0.9, 30, {"estimator": "rf_3"}),
        ]

        # Create repeat results
        repeat_1 = RepeatResult.from_fold_results(0, fold_results_1)
        repeat_2 = RepeatResult.from_fold_results(1, fold_results_2)

        # Create evaluation result
        eval_result = EvaluationResult.from_repeat_results(
            model_name="integration_test", model_format="mc", metric_name="acc", repeat_results=[repeat_1, repeat_2]
        )

        # Verify complete pipeline
        assert len(eval_result.repeat_results) == 2
        assert eval_result.repeat_results[0].total_instances == 90  # 3 folds * 30
        assert eval_result.repeat_results[1].total_instances == 90
        assert eval_result.total_count == 90  # From first repeat

        # Check that metadata is preserved
        assert eval_result.repeat_results[0].fold_results[0].metadata["estimator"] == "rf_1"

    def test_result_consistency(self):
        """Test that results are mathematically consistent"""
        # Create known scores
        scores = [0.6, 0.8, 1.0]
        fold_results = [FoldResult(i + 1, score, 20) for i, score in enumerate(scores)]

        # Create repeat result
        repeat_result = RepeatResult.from_fold_results(0, fold_results)

        # Verify statistics
        expected_mean = np.mean(scores)
        expected_std = np.std(scores)

        assert abs(repeat_result.mean_score - expected_mean) < 1e-10
        assert abs(repeat_result.std_score - expected_std) < 1e-10

        # Create evaluation result
        eval_result = EvaluationResult.from_repeat_results(
            model_name="consistency_test", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

        # Overall statistics should match repeat statistics for single repeat
        assert abs(eval_result.overall_mean - expected_mean) < 1e-10
        assert abs(eval_result.overall_std - 0.0) < 1e-10  # Single repeat -> std = 0

    def test_large_scale_results(self):
        """Test results with many folds and repeats"""
        # Create 10 repeats with 5 folds each
        repeat_results = []

        for repeat_id in range(10):
            fold_results = []
            for fold_id in range(5):
                # Use slightly different scores for variety
                score = 0.7 + 0.1 * np.sin(repeat_id + fold_id)
                fold_results.append(FoldResult(fold_id + 1, score, 50))

            repeat_results.append(RepeatResult.from_fold_results(repeat_id, fold_results))

        # Create evaluation result
        eval_result = EvaluationResult.from_repeat_results(
            model_name="large_scale_test", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Verify structure
        assert len(eval_result.repeat_results) == 10
        assert eval_result.total_count == 250  # 5 folds * 50 samples
        assert 0.0 <= eval_result.overall_mean <= 1.0
        assert eval_result.overall_std >= 0.0
