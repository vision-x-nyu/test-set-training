"""
Tests for unified evaluation result objects.

These tests ensure the result data structures work correctly and provide
expected functionality for aggregating and converting evaluation results.
"""

import pytest
import numpy as np

from TsT.core.protocols import FoldResult, RepeatResult, EvaluationResult


@pytest.fixture
def sample_fold_results():
    """Create sample fold results for testing"""
    return [
        FoldResult(1, 0.8, 20, list(np.random.randint(0, 100, 20)), "acc"),
        FoldResult(2, 0.9, 15, list(np.random.randint(0, 100, 15)), "acc"),
        FoldResult(3, 0.7, 25, list(np.random.randint(0, 100, 25)), "acc"),
    ]


@pytest.fixture
def sample_repeat_results():
    """Create sample repeat results for testing"""
    fold_results = [
        FoldResult(1, 0.8, 20, list(np.random.randint(0, 100, 20)), "acc"),
        FoldResult(2, 0.9, 15, list(np.random.randint(0, 100, 15)), "acc"),
        FoldResult(3, 0.7, 25, list(np.random.randint(0, 100, 25)), "acc"),
    ]
    return [
        RepeatResult.from_fold_results(0, fold_results),
        RepeatResult.from_fold_results(1, fold_results),
    ]


class TestFoldResult:
    """Test FoldResult functionality"""

    def test_basic_creation(self):
        """Test basic FoldResult creation"""
        test_idx = list(np.random.randint(0, 100, 20))
        result = FoldResult(
            fold_id=1,
            score=0.85,
            train_size=20,
            test_idx=test_idx,
            metric="acc",
            metadata={"test": True},
        )

        assert result.fold_id == 1
        assert np.isclose(result.score, 0.85, atol=1e-6), result.score
        assert result.train_size == 20
        assert result.test_size == 20
        assert result.test_idx == test_idx
        assert result.metadata["test"] is True

    def test_default_metadata(self):
        """Test FoldResult with default metadata"""
        test_idx = list(np.random.randint(0, 100, 15))
        result = FoldResult(fold_id=2, score=0.7, train_size=15, test_idx=test_idx, metric="acc")

        assert result.metadata == {}
        assert isinstance(result.metadata, dict)

    def test_metadata_modification(self):
        """Test that metadata can be modified after creation"""
        test_idx = list(np.random.randint(0, 100, 10))
        result = FoldResult(fold_id=1, score=0.8, train_size=10, test_idx=test_idx, metric="acc")
        result.metadata["new_key"] = "new_value"

        assert result.metadata["new_key"] == "new_value"


class TestRepeatResult:
    """Test RepeatResult functionality"""

    def create_sample_fold_results(self):
        """Create sample fold results for testing"""
        return [
            FoldResult(1, 0.8, 20, list(np.random.randint(0, 100, 20)), "acc"),
            FoldResult(2, 0.9, 15, list(np.random.randint(0, 100, 15)), "acc"),
            FoldResult(3, 0.7, 25, list(np.random.randint(0, 100, 25)), "acc"),
        ]

    def test_basic_creation(self):
        """Test basic RepeatResult creation"""
        fold_results = self.create_sample_fold_results()

        result = RepeatResult(repeat_id=0, fold_results=fold_results, mean_score=0.8, std_score=0.1)

        assert result.repeat_id == 0
        assert len(result.fold_results) == 3
        assert np.isclose(result.mean_score, 0.8, atol=1e-6), result.mean_score
        assert np.isclose(result.std_score, 0.1, atol=1e-6), result.std_score

    def test_total_instances(self):
        """Test total_instances property"""
        fold_results = self.create_sample_fold_results()

        result = RepeatResult(0, fold_results, 0.8, 0.1)
        assert result.total_instances == 60  # 20 + 15 + 25

    def test_from_fold_results(self):
        """Test creating RepeatResult from fold results with calculated statistics"""
        fold_results = self.create_sample_fold_results()

        result = RepeatResult.from_fold_results(0, fold_results)

        assert result.repeat_id == 0
        assert len(result.fold_results) == 3

        # Check calculated statistics using weighted mean/std
        scores = [fold.score for fold in fold_results]
        counts = [fold.test_size for fold in fold_results]
        expected_mean = np.average(scores, weights=counts)
        expected_std = np.sqrt(np.average((scores - expected_mean) ** 2, weights=counts))

        assert np.isclose(result.mean_score, expected_mean, atol=1e-6), result.mean_score
        assert np.isclose(result.std_score, expected_std, atol=1e-6), result.std_score

    def test_empty_fold_results(self):
        """Test RepeatResult with empty fold results"""
        result = RepeatResult.from_fold_results(0, [])

        assert result.repeat_id == 0
        assert len(result.fold_results) == 0
        assert result.total_instances == 0
        # Mean and std of empty list should be 0
        assert np.isclose(result.mean_score, 0.0, atol=1e-6), result.mean_score
        assert np.isclose(result.std_score, 0.0, atol=1e-6), result.std_score


class TestEvaluationResult:
    """Test EvaluationResult functionality"""

    def create_sample_repeat_results(self):
        """Create sample repeat results for testing"""
        # First repeat
        fold_results_1 = [
            FoldResult(1, 0.8, 25, list(np.random.randint(0, 100, 25)), "acc"),
            FoldResult(2, 0.9, 25, list(np.random.randint(0, 100, 25)), "acc"),
        ]
        repeat_1 = RepeatResult.from_fold_results(0, fold_results_1)

        # Second repeat
        fold_results_2 = [
            FoldResult(1, 0.7, 25, list(np.random.randint(0, 100, 25)), "acc"),
            FoldResult(2, 0.8, 25, list(np.random.randint(0, 100, 25)), "acc"),
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
            count=100,
            repeats=2,
            total_count=200,
        )

        assert result.model_name == "test_model"
        assert result.model_format == "mc"
        assert result.metric_name == "acc"
        assert len(result.repeat_results) == 2
        assert np.isclose(result.overall_mean, 0.825, atol=1e-6), result.overall_mean
        assert np.isclose(result.overall_std, 0.025, atol=1e-6), result.overall_std
        assert result.count == 100
        assert result.total_count == 100 * 2  # 100 samples * 2 repeats

    def test_from_repeat_results(self):
        """Test creating EvaluationResult from repeat results with calculated statistics"""
        repeat_results = self.create_sample_repeat_results()

        result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Check calculated statistics using weighted mean/std across all folds
        # All folds have equal weight (25 samples each)
        # Overall mean: (0.8 + 0.9 + 0.7 + 0.8) / 4 = 0.8
        # Overall std: weighted std of [0.8, 0.9, 0.7, 0.8] with weights [25, 25, 25, 25]
        flat_scores = [0.8, 0.9, 0.7, 0.8]
        flat_counts = [25, 25, 25, 25]
        expected_mean = np.average(flat_scores, weights=flat_counts)
        expected_std = np.sqrt(np.average((flat_scores - expected_mean) ** 2, weights=flat_counts))

        assert np.isclose(result.overall_mean, expected_mean, atol=1e-4), result.overall_mean
        assert np.isclose(result.overall_std, expected_std, atol=1e-4), result.overall_std
        assert result.count == 25 * 2  # 25 * 2 folds
        assert result.total_count == 25 * 2 * 2  # 25 * 2 folds * 2 repeats

    def test_to_from_dict(self):
        """Test conversion to dictionary and back using from_dict method"""

        repeat_results = self.create_sample_repeat_results()

        original_result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Convert to dictionary (as would happen with JSON serialization)
        summary = original_result.to_dict()

        # Verify dictionary structure
        assert summary["model_name"] == "test_model"
        assert summary["model_format"] == "mc"
        assert summary["metric_name"] == "acc"
        assert np.isclose(summary["overall_mean"], 0.8, atol=1e-4), summary["overall_mean"]
        assert np.isclose(summary["overall_std"], 0.0707, atol=1e-4), summary["overall_std"]
        assert summary["count"] == 25 * 2
        assert summary["repeats"] == 2
        assert summary["total_count"] == 25 * 2 * 2

        # Reconstruct from dictionary using from_dict method
        reconstructed_result = EvaluationResult.from_dict(summary)

        # Verify the reconstructed object matches the original
        assert isinstance(reconstructed_result, EvaluationResult)
        assert reconstructed_result.model_name == original_result.model_name
        assert reconstructed_result.model_format == original_result.model_format
        assert reconstructed_result.metric_name == original_result.metric_name
        assert np.isclose(reconstructed_result.overall_mean, original_result.overall_mean, atol=1e-10)
        assert np.isclose(reconstructed_result.overall_std, original_result.overall_std, atol=1e-10)
        assert reconstructed_result.count == original_result.count
        assert reconstructed_result.repeats == original_result.repeats
        assert reconstructed_result.total_count == original_result.total_count

        # Verify nested objects are properly reconstructed
        assert len(reconstructed_result.repeat_results) == len(original_result.repeat_results)
        for i, (orig_repeat, recon_repeat) in enumerate(
            zip(original_result.repeat_results, reconstructed_result.repeat_results)
        ):
            assert recon_repeat.repeat_id == orig_repeat.repeat_id
            assert np.isclose(recon_repeat.mean_score, orig_repeat.mean_score, atol=1e-10)
            assert np.isclose(recon_repeat.std_score, orig_repeat.std_score, atol=1e-10)
            assert len(recon_repeat.fold_results) == len(orig_repeat.fold_results)

            for j, (orig_fold, recon_fold) in enumerate(zip(orig_repeat.fold_results, recon_repeat.fold_results)):
                assert recon_fold.fold_id == orig_fold.fold_id
                assert np.isclose(recon_fold.score, orig_fold.score, atol=1e-10)
                assert recon_fold.train_size == orig_fold.train_size
                assert recon_fold.test_idx == orig_fold.test_idx
                assert recon_fold.metric == orig_fold.metric
                assert recon_fold.metadata == orig_fold.metadata

    def test_to_json(self):
        """Test the to_json() method"""
        repeat_results = self.create_sample_repeat_results()

        original_result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Use the to_json() method
        json_str = original_result.to_json()

        # Verify it's valid JSON
        import json

        parsed_dict = json.loads(json_str)

        # Verify the JSON contains expected fields
        assert parsed_dict["model_name"] == "test_model"
        assert parsed_dict["model_format"] == "mc"
        assert parsed_dict["metric_name"] == "acc"
        assert "overall_mean" in parsed_dict
        assert "overall_std" in parsed_dict
        assert "repeat_results" in parsed_dict
        assert len(parsed_dict["repeat_results"]) == 2

    def test_json_serialization_roundtrip(self):
        """Test full JSON serialization and deserialization roundtrip using to_json() and from_dict()"""
        import json

        repeat_results = self.create_sample_repeat_results()

        original_result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Use the to_json() method for serialization
        json_str = original_result.to_json()

        # Parse JSON back to dictionary
        parsed_dict = json.loads(json_str)

        # Reconstruct from dictionary using from_dict()
        reconstructed_result = EvaluationResult.from_dict(parsed_dict)

        # Verify the reconstructed object matches the original
        assert reconstructed_result.model_name == original_result.model_name
        assert reconstructed_result.model_format == original_result.model_format
        assert reconstructed_result.metric_name == original_result.metric_name
        assert np.isclose(reconstructed_result.overall_mean, original_result.overall_mean, atol=1e-10)
        assert np.isclose(reconstructed_result.overall_std, original_result.overall_std, atol=1e-10)
        assert reconstructed_result.count == original_result.count
        assert reconstructed_result.repeats == original_result.repeats
        assert reconstructed_result.total_count == original_result.total_count

        # Verify nested structure is preserved
        assert len(reconstructed_result.repeat_results) == len(original_result.repeat_results)
        for orig_repeat, recon_repeat in zip(original_result.repeat_results, reconstructed_result.repeat_results):
            assert recon_repeat.repeat_id == orig_repeat.repeat_id
            assert len(recon_repeat.fold_results) == len(orig_repeat.fold_results)
            for orig_fold, recon_fold in zip(orig_repeat.fold_results, recon_repeat.fold_results):
                assert recon_fold.fold_id == orig_fold.fold_id
                assert np.isclose(recon_fold.score, orig_fold.score, atol=1e-10)
                assert recon_fold.test_idx == orig_fold.test_idx

    def test_save_and_load_file(self):
        """Test saving and loading EvaluationResult to/from file"""
        import tempfile
        import os

        repeat_results = self.create_sample_repeat_results()

        original_result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            tmp_filepath = tmp_file.name

        try:
            # Save to file
            original_result.save_to_file(tmp_filepath)

            # Verify file was created
            assert os.path.exists(tmp_filepath)

            # Load from file
            loaded_result = EvaluationResult.load_from_file(tmp_filepath)

            # Verify the loaded object matches the original
            assert loaded_result.model_name == original_result.model_name
            assert loaded_result.model_format == original_result.model_format
            assert loaded_result.metric_name == original_result.metric_name
            assert np.isclose(loaded_result.overall_mean, original_result.overall_mean, atol=1e-10)
            assert np.isclose(loaded_result.overall_std, original_result.overall_std, atol=1e-10)
            assert loaded_result.count == original_result.count
            assert loaded_result.repeats == original_result.repeats
            assert loaded_result.total_count == original_result.total_count

            # Verify nested structure is preserved
            assert len(loaded_result.repeat_results) == len(original_result.repeat_results)
            for orig_repeat, loaded_repeat in zip(original_result.repeat_results, loaded_result.repeat_results):
                assert loaded_repeat.repeat_id == orig_repeat.repeat_id
                assert len(loaded_repeat.fold_results) == len(orig_repeat.fold_results)
                for orig_fold, loaded_fold in zip(orig_repeat.fold_results, loaded_repeat.fold_results):
                    assert loaded_fold.fold_id == orig_fold.fold_id
                    assert np.isclose(loaded_fold.score, orig_fold.score, atol=1e-10)
                    assert loaded_fold.test_idx == orig_fold.test_idx

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filepath):
                os.unlink(tmp_filepath)

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

    def test_fold_scores_per_repeat_property(self):
        """Test fold_scores_per_repeat property"""
        repeat_results = self.create_sample_repeat_results()

        result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        fold_scores = result.fold_scores_per_repeat

        # Should be shape (2 repeats, 2 folds)
        assert fold_scores.shape == (2, 2)

        # Check values match expected scores
        expected_scores = np.array([[0.8, 0.9], [0.7, 0.8]])
        np.testing.assert_array_equal(fold_scores, expected_scores)

    def test_repeat_scores_property(self):
        """Test repeat_scores property"""
        repeat_results = self.create_sample_repeat_results()

        result = EvaluationResult.from_repeat_results(
            model_name="test_model", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        repeat_scores = result.repeat_scores

        # Should be shape (2 repeats,)
        assert repeat_scores.shape == (2,)

        # Check values match expected mean scores
        expected_scores = np.array([repeat.mean_score for repeat in repeat_results])
        np.testing.assert_array_equal(repeat_scores, expected_scores)

    def test_empty_repeat_results(self):
        """Test EvaluationResult with empty repeat results should raise AssertionError"""

        with pytest.raises(AssertionError, match="at least one repeat"):
            EvaluationResult.from_repeat_results(
                model_name="empty_model", model_format="mc", metric_name="acc", repeat_results=[]
            )


class TestResultsIntegration:
    """Test integration between different result types"""

    def test_complete_result_pipeline(self):
        """Test complete pipeline from fold results to evaluation result"""
        # Create fold results for two repeats
        fold_results_1 = [
            FoldResult(1, 0.9, 30, list(np.random.randint(0, 100, 30)), "acc", {"estimator": "rf_1"}),
            FoldResult(2, 0.8, 30, list(np.random.randint(0, 100, 30)), "acc", {"estimator": "rf_2"}),
            FoldResult(3, 0.85, 30, list(np.random.randint(0, 100, 30)), "acc", {"estimator": "rf_3"}),
        ]

        fold_results_2 = [
            FoldResult(1, 0.95, 30, list(np.random.randint(0, 100, 30)), "acc", {"estimator": "rf_1"}),
            FoldResult(2, 0.75, 30, list(np.random.randint(0, 100, 30)), "acc", {"estimator": "rf_2"}),
            FoldResult(3, 0.9, 30, list(np.random.randint(0, 100, 30)), "acc", {"estimator": "rf_3"}),
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
        assert eval_result.count == 90  # From first repeat
        assert eval_result.total_count == 90 * 2  # 90 samples * 2 repeats

        # Check that metadata is preserved
        assert eval_result.repeat_results[0].fold_results[0].metadata["estimator"] == "rf_1"

    def test_result_consistency(self):
        """Test that results are mathematically consistent"""
        # Create known scores
        scores = [0.6, 0.8, 1.0]
        fold_results = [
            FoldResult(i + 1, score, 20, list(np.random.randint(0, 100, 20)), "acc") for i, score in enumerate(scores)
        ]

        # Create repeat result
        repeat_result = RepeatResult.from_fold_results(0, fold_results)

        # Verify statistics
        expected_mean = np.mean(scores)
        expected_std = np.std(scores)

        assert np.isclose(repeat_result.mean_score, expected_mean, atol=1e-10), repeat_result.mean_score
        assert np.isclose(repeat_result.std_score, expected_std, atol=1e-10), repeat_result.std_score

        # Create evaluation result
        eval_result = EvaluationResult.from_repeat_results(
            model_name="consistency_test", model_format="mc", metric_name="acc", repeat_results=[repeat_result]
        )

        # Overall statistics should match repeat statistics for single repeat
        assert np.isclose(eval_result.overall_mean, expected_mean, atol=1e-10), eval_result.overall_mean
        assert np.isclose(eval_result.overall_std, expected_std, atol=1e-10), eval_result.overall_std

    def test_large_scale_results(self):
        """Test results with many folds and repeats"""
        # Create 10 repeats with 5 folds each
        repeat_results = []

        for repeat_id in range(10):
            fold_results = []
            for fold_id in range(5):
                # Use slightly different scores for variety
                score = 0.7 + 0.1 * np.sin(repeat_id + fold_id)
                fold_results.append(FoldResult(fold_id + 1, score, 50, list(np.random.randint(0, 100, 50)), "acc"))

            repeat_results.append(RepeatResult.from_fold_results(repeat_id, fold_results))

        # Create evaluation result
        eval_result = EvaluationResult.from_repeat_results(
            model_name="large_scale_test", model_format="mc", metric_name="acc", repeat_results=repeat_results
        )

        # Verify structure
        assert len(eval_result.repeat_results) == 10
        assert eval_result.count == 250  # 5 folds * 50 samples
        assert eval_result.total_count == 250 * 10  # 250 samples * 10 repeats
        assert 0.0 <= eval_result.overall_mean <= 1.0
        assert eval_result.overall_std >= 0.0
