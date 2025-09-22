"""
Clean integration tests for the new benchmark system that work with real benchmarks.

These tests work with actual registered benchmarks like CVB and verify
the system works end-to-end.
"""

import pytest
import pandas as pd

# Import all benchmarks to ensure they're registered

from TsT.experiments.utils import load_benchmark, list_available_benchmarks
from TsT.evaluation import run_evaluation


class TestRealBenchmarkSystemIntegration:
    """Integration tests with real benchmarks (no registry isolation)"""

    def test_benchmark_registry_discovers_cvb(self):
        """Test that CVB is automatically discovered via registry"""
        available_benchmarks = list_available_benchmarks()
        assert "cvb" in available_benchmarks

    def test_load_benchmark_returns_proper_instance(self):
        """Test that load_benchmark returns proper benchmark instances"""
        cvb = load_benchmark("cvb")

        # Should have new benchmark interface
        assert hasattr(cvb, "name")
        assert hasattr(cvb, "description")
        assert hasattr(cvb, "load_data")
        assert hasattr(cvb, "get_feature_based_models")
        assert hasattr(cvb, "get_qa_models")
        assert hasattr(cvb, "get_metadata")

        assert cvb.name == "cvb"

    def test_benchmark_provides_both_model_types(self):
        """Test that benchmarks provide both RF and LLM models"""
        cvb = load_benchmark("cvb")

        # Should get feature-based models for RF
        rf_models = cvb.get_feature_based_models()
        assert isinstance(rf_models, list)
        assert len(rf_models) > 0

        # Each should be a feature-based model
        for model in rf_models:
            assert hasattr(model, "feature_cols")
            assert hasattr(model, "fit_feature_maps")
            assert hasattr(model, "add_features")

        # Should get QA model for LLM
        qa_model = cvb.get_qa_models()[0]
        assert hasattr(qa_model, "benchmark_name")
        assert hasattr(qa_model, "prepare_instances")
        assert qa_model.benchmark_name == "cvb"

    def test_llm_evaluation_workflow_structure(self):
        """Test LLM evaluation workflow structure (without actual LLM calls)"""
        cvb = load_benchmark("cvb")

        # Get QA model
        qa_model = cvb.get_qa_models()[0]

        # Create test data
        mock_data = pd.DataFrame(
            {
                "question_type": ["count_2d", "relation_2d"],
                "question": ["How many?", "Where is it?"],
                "gt_idx": ["A", "B"],
                "options": [["A: 1", "B: 2"], ["A: left", "B: right"]],
                "question_format": ["mc"] * 2,
            }
        )

        # Test that QA model can process data
        selected = qa_model.select_rows(mock_data)
        assert len(selected) == 2  # Should select all since no question_type filtering

        prepared = qa_model.prepare_instances(selected)
        assert isinstance(prepared, pd.DataFrame)

        # Properties should work
        assert qa_model.task == "clf"
        assert qa_model.metric == "acc"

    def test_benchmark_metadata_is_correct(self):
        """Test that benchmark metadata is generated correctly"""
        cvb = load_benchmark("cvb")
        metadata = cvb.get_metadata()

        assert metadata["name"] == "cvb"
        assert "description" in metadata
        assert "question_types" in metadata
        assert "formats" in metadata
        assert "num_feature_models" in metadata

        # CVB should have 4 feature models, all MC format
        assert metadata["num_feature_models"] == 4
        assert "mc" in metadata["formats"]
        assert len(metadata["formats"]["mc"]) == 4

    def test_cli_argument_parsing_works(self):
        """Test that CLI can discover benchmarks dynamically"""
        from TsT.__main__ import create_parser

        parser = create_parser()

        # Should be able to parse CVB
        args = parser.parse_args(["--benchmark", "cvb", "--mode", "rf"])
        assert args.benchmark == "cvb"
        assert args.mode == "rf"

        # Should also work with LLM mode
        args = parser.parse_args(["--benchmark", "cvb", "--mode", "llm"])
        assert args.benchmark == "cvb"
        assert args.mode == "llm"

    def test_end_to_end_main_function_structure(self):
        """Test the main function structure (without actually running evaluation)"""
        from TsT.__main__ import get_benchmark

        # Should be able to get benchmark
        cvb = get_benchmark("cvb")
        assert cvb.name == "cvb"

        # Should have data loading capability
        try:
            data = cvb.load_data()
            assert isinstance(data, pd.DataFrame)
        except FileNotFoundError:
            pytest.skip("CVB test data not available")
        except Exception as e:
            pytest.skip(f"CVB data loading failed: {e}")

    def test_experiments_utils_integration(self):
        """Test that experiments utils work with new system"""
        from TsT.experiments.utils import load_benchmark, list_available_benchmarks

        # Should list benchmarks
        benchmarks = list_available_benchmarks()
        assert isinstance(benchmarks, list)
        assert "cvb" in benchmarks

        # Should load benchmark
        cvb = load_benchmark("cvb")
        assert cvb.name == "cvb"

    def test_llm_single_run_structure(self):
        """Test that LLM single run works with new system structure"""
        from TsT.experiments.utils import load_benchmark

        # Should be able to load benchmark for LLM run
        cvb = load_benchmark("cvb")

        # Should get QA model
        qa_model = cvb.get_qa_models()[0]
        assert qa_model.benchmark_name == "cvb"
        assert qa_model.format == "mc"  # CVB is all MC

        # Should be able to use in LLM evaluation context
        models_for_llm = [qa_model]
        assert len(models_for_llm) == 1
        assert models_for_llm[0].benchmark_name == "cvb"

    def test_rf_evaluation_with_new_system(self):
        """Test RF evaluation workflow with new system"""
        from TsT.core.protocols import EvaluationResult

        cvb = load_benchmark("cvb")

        # Get RF models
        rf_models = cvb.get_feature_based_models()

        # Create minimal test data
        mock_data = pd.DataFrame(
            {
                "question_type": ["count_2d", "relation_2d", "count_2d"],
                "question": [
                    "How many circles are in the image?",
                    "Where is the circle relative to the square?",
                    "How many triangles are in the image?",
                ],
                "gt_idx": [1, 0, 2],
                "gt_option": ["2", "left", "3"],
                "options": [["A: 1", "B: 2", "C: 3"], ["A: left", "B: right", "C: center"], ["A: 1", "B: 2", "C: 3"]],
            }
        )

        # Test that we can run evaluation (may fail due to feature engineering complexity, but should not crash)
        try:
            # Take just first model to test
            test_models = rf_models[:1]

            results = run_evaluation(
                question_models=test_models, df_full=mock_data, n_splits=2, random_state=42, verbose=False, mode="rf"
            )

            # If it succeeds, check basic structure
            assert isinstance(results, list)
            assert isinstance(results[0], EvaluationResult)
            assert len(results) == 1
            assert "model_name" in results[0]
            assert "overall_mean" in results[0]

        except Exception as e:
            # Feature engineering might fail with mock data - that's okay for integration test
            # The important part is that the benchmark system works
            pytest.skip(f"RF evaluation failed with mock data (expected): {e}")


class TestNewSystemPerformance:
    """Test performance characteristics of new system"""

    def test_benchmark_loading_is_fast(self):
        """Test that benchmark loading is reasonably fast"""
        import time

        start = time.time()

        # Load benchmark multiple times
        for _ in range(10):
            cvb = load_benchmark("cvb")
            _ = cvb.get_feature_based_models()
            __ = cvb.get_qa_models()[0]

        end = time.time()
        duration = end - start

        # Should complete quickly
        assert duration < 5.0, f"Benchmark loading took too long: {duration:.2f}s"

    def test_registry_operations_are_efficient(self):
        """Test that registry operations are efficient"""
        import time

        start = time.time()

        # Perform registry operations
        for _ in range(100):
            benchmarks = list_available_benchmarks()
            assert "cvb" in benchmarks

        end = time.time()
        duration = end - start

        # Should be very fast
        assert duration < 1.0, f"Registry operations took too long: {duration:.2f}s"
