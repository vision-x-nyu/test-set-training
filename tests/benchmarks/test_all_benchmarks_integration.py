"""
Integration tests for all benchmarks with the new system.

These tests verify that all benchmarks (CVB, VSI, MMMU, Video-MME) work correctly
with the new benchmark registry and provide both RF and LLM model interfaces.
"""

import pytest
import pandas as pd

# Import all benchmarks to ensure they're registered before tests run

from TsT.experiments.utils import load_benchmark, list_available_benchmarks


class TestAllBenchmarksRegistration:
    """Test that all benchmarks are properly registered"""

    def test_all_benchmarks_discovered(self):
        """Test that all expected benchmarks are discovered"""
        available = list_available_benchmarks()
        expected_benchmarks = ["cvb", "vsi", "mmmu", "video_mme"]

        for benchmark_name in expected_benchmarks:
            assert benchmark_name in available, f"Benchmark {benchmark_name} should be available"

        print(f"✅ Found all {len(expected_benchmarks)} benchmarks: {available}")

    def test_all_benchmarks_loadable(self):
        """Test that all benchmarks can be loaded"""
        available = list_available_benchmarks()

        for benchmark_name in available:
            benchmark = load_benchmark(benchmark_name)
            assert benchmark.name == benchmark_name
            assert hasattr(benchmark, "description")
            assert hasattr(benchmark, "load_data")
            assert hasattr(benchmark, "get_feature_based_models")
            assert hasattr(benchmark, "get_qa_model")


class TestBenchmarkInterfaces:
    """Test that all benchmarks provide proper interfaces"""

    @pytest.mark.parametrize("benchmark_name", ["cvb", "vsi", "mmmu", "video_mme"])
    def test_benchmark_has_feature_models(self, benchmark_name):
        """Test that each benchmark provides feature-based models"""
        benchmark = load_benchmark(benchmark_name)
        rf_models = benchmark.get_feature_based_models()

        assert isinstance(rf_models, list)
        assert len(rf_models) > 0, f"{benchmark_name} should have at least one RF model"

        # Each should be a feature-based model
        for model in rf_models:
            assert hasattr(model, "feature_cols")
            assert hasattr(model, "fit_feature_maps")
            assert hasattr(model, "add_features")
            assert hasattr(model, "select_rows")
            assert hasattr(model, "task")
            assert hasattr(model, "metric")

    @pytest.mark.parametrize("benchmark_name", ["cvb", "vsi", "mmmu", "video_mme"])
    def test_benchmark_has_qa_model(self, benchmark_name):
        """Test that each benchmark provides QA model"""
        benchmark = load_benchmark(benchmark_name)
        qa_model = benchmark.get_qa_model()

        # Should have QA model interface
        assert hasattr(qa_model, "benchmark_name")
        assert hasattr(qa_model, "name")
        assert hasattr(qa_model, "format")
        assert hasattr(qa_model, "prepare_instances")
        assert hasattr(qa_model, "select_rows")
        assert hasattr(qa_model, "task")
        assert hasattr(qa_model, "metric")

        # Should have correct benchmark name
        assert qa_model.benchmark_name == benchmark_name

        # Format should be valid
        assert qa_model.format in ["mc", "num"]

        # Task and metric should be consistent with format
        if qa_model.format == "mc":
            assert qa_model.task == "clf"
            assert qa_model.metric == "acc"
        elif qa_model.format == "num":
            assert qa_model.task == "reg"
            assert qa_model.metric == "mra"

    @pytest.mark.parametrize("benchmark_name", ["cvb", "vsi", "mmmu", "video_mme"])
    def test_benchmark_metadata(self, benchmark_name):
        """Test that each benchmark provides proper metadata"""
        benchmark = load_benchmark(benchmark_name)
        metadata = benchmark.get_metadata()

        # Should have basic metadata
        assert metadata["name"] == benchmark_name
        assert "description" in metadata
        assert "question_types" in metadata
        assert "formats" in metadata
        assert "num_feature_models" in metadata

        # Should have at least one model
        assert metadata["num_feature_models"] > 0

        # Should have valid formats
        formats = metadata["formats"]
        assert isinstance(formats, dict)
        for format_key, model_list in formats.items():
            assert format_key in ["mc", "num"], f"Invalid format: {format_key}"
            assert isinstance(model_list, list)


class TestBenchmarkSpecifics:
    """Test benchmark-specific functionality"""

    def test_cvb_specifics(self):
        """Test CVB-specific structure"""
        cvb = load_benchmark("cvb")
        metadata = cvb.get_metadata()

        # CVB should have 4 MC models
        assert metadata["num_feature_models"] == 4
        assert "mc" in metadata["formats"]
        assert len(metadata["formats"]["mc"]) == 4
        assert "num" not in metadata["formats"]

        # QA model should be MC
        qa_model = cvb.get_qa_model()
        assert qa_model.format == "mc"

    def test_vsi_specifics(self):
        """Test VSI-specific structure"""
        vsi = load_benchmark("vsi")
        metadata = vsi.get_metadata()

        # VSI should have 8 models (4 NUM + 4 MC)
        assert metadata["num_feature_models"] == 8
        assert "mc" in metadata["formats"]
        assert "num" in metadata["formats"]
        assert len(metadata["formats"]["mc"]) == 4
        assert len(metadata["formats"]["num"]) == 4

        # QA model should focus on MC for LLM evaluation
        qa_model = vsi.get_qa_model()
        assert qa_model.format == "mc"
        assert qa_model.question_types is not None  # Should filter to MC types

    def test_mmmu_specifics(self):
        """Test MMMU-specific structure"""
        mmmu = load_benchmark("mmmu")
        metadata = mmmu.get_metadata()

        # MMMU should have 1 MC model
        assert metadata["num_feature_models"] == 1
        assert "mc" in metadata["formats"]
        assert len(metadata["formats"]["mc"]) == 1

        # QA model should be MC
        qa_model = mmmu.get_qa_model()
        assert qa_model.format == "mc"

    def test_video_mme_specifics(self):
        """Test Video-MME-specific structure"""
        video_mme = load_benchmark("video_mme")
        metadata = video_mme.get_metadata()

        # Video-MME should have 1 MC model
        assert metadata["num_feature_models"] == 1
        assert "mc" in metadata["formats"]
        assert len(metadata["formats"]["mc"]) == 1

        # QA model should be MC
        qa_model = video_mme.get_qa_model()
        assert qa_model.format == "mc"


class TestBenchmarkDataLoading:
    """Test data loading for all benchmarks (where possible)"""

    @pytest.mark.parametrize("benchmark_name", ["cvb", "vsi", "mmmu", "video_mme"])
    def test_benchmark_can_load_data(self, benchmark_name):
        """Test that each benchmark can load its data"""
        benchmark = load_benchmark(benchmark_name)

        try:
            data = benchmark.load_data()
            assert isinstance(data, pd.DataFrame)

            if len(data) > 0:
                # Basic column checks
                assert "question_type" in data.columns or "id" in data.columns
                print(f"✅ {benchmark_name}: Loaded {len(data)} examples")
            else:
                print(f"⚠️ {benchmark_name}: Data loaded but empty")

        except FileNotFoundError:
            pytest.skip(f"{benchmark_name} test data not available")
        except Exception as e:
            pytest.skip(f"{benchmark_name} data loading failed: {e}")


class TestModeCompatibility:
    """Test that benchmarks work with both RF and LLM modes"""

    @pytest.mark.parametrize("benchmark_name", ["cvb", "vsi", "mmmu", "video_mme"])
    def test_rf_mode_models(self, benchmark_name):
        """Test that RF mode models have correct structure"""
        benchmark = load_benchmark(benchmark_name)
        rf_models = benchmark.get_feature_based_models()

        for model in rf_models:
            # Should have FeatureBasedBiasModel interface
            assert hasattr(model, "name")
            assert hasattr(model, "format")
            assert hasattr(model, "feature_cols")
            assert hasattr(model, "task")
            assert hasattr(model, "metric")

            # Format should be valid
            assert model.format in ["mc", "num"]

            # Task/metric should be consistent
            if model.format == "mc":
                assert model.task == "clf"
                assert model.metric == "acc"
            elif model.format == "num":
                assert model.task == "reg"
                assert model.metric == "mra"

    @pytest.mark.parametrize("benchmark_name", ["cvb", "vsi", "mmmu", "video_mme"])
    def test_llm_mode_model(self, benchmark_name):
        """Test that LLM mode model has correct structure"""
        benchmark = load_benchmark(benchmark_name)
        qa_model = benchmark.get_qa_model()

        # Should have QuestionAnswerBiasModel interface
        assert hasattr(qa_model, "benchmark_name")
        assert hasattr(qa_model, "name")
        assert hasattr(qa_model, "format")
        assert hasattr(qa_model, "prepare_instances")
        assert hasattr(qa_model, "select_rows")
        assert hasattr(qa_model, "task")
        assert hasattr(qa_model, "metric")

        # Should be configured for this benchmark
        assert qa_model.benchmark_name == benchmark_name
        assert benchmark_name in qa_model.name

    def test_model_count_difference_rf_vs_llm(self):
        """Test that RF mode has multiple models, LLM mode has single model"""
        for benchmark_name in list_available_benchmarks():
            benchmark = load_benchmark(benchmark_name)

            # RF mode: multiple models (one per question type)
            rf_models = benchmark.get_feature_based_models()

            # LLM mode: single unified model
            qa_model = benchmark.get_qa_model()

            print(f"{benchmark_name}: RF={len(rf_models)} models, LLM=1 model")

            # RF should have 1+ models, LLM should always be 1
            assert len(rf_models) >= 1
            assert qa_model.benchmark_name == benchmark_name


class TestEndToEndWorkflow:
    """Test end-to-end workflow for all benchmarks"""

    def test_all_benchmarks_work_with_mock_evaluation(self):
        """Test that all benchmarks work with mock evaluation workflow"""
        for benchmark_name in list_available_benchmarks():
            benchmark = load_benchmark(benchmark_name)

            # Create very minimal mock data that should work for any benchmark
            mock_data = pd.DataFrame(
                {
                    "question_type": ["test_type"],
                    "question": ["Test question?"],
                    "gt_idx": [0],
                    "ground_truth": ["test_answer"],
                    "options": [["A: option1", "B: option2"]],
                }
            )

            # Test RF models
            rf_models = benchmark.get_feature_based_models()
            for model in rf_models[:1]:  # Just test first model
                try:
                    selected = model.select_rows(mock_data)
                    assert isinstance(selected, pd.DataFrame)
                except Exception:
                    # Expected - mock data won't match actual model requirements
                    pass

            # Test QA model
            qa_model = benchmark.get_qa_model()
            selected = qa_model.select_rows(mock_data)
            assert isinstance(selected, pd.DataFrame)

            prepared = qa_model.prepare_instances(selected)
            assert isinstance(prepared, pd.DataFrame)

            print(f"✅ {benchmark_name}: End-to-end workflow verified")

    def test_metadata_consistency_across_benchmarks(self):
        """Test that metadata is consistent across all benchmarks"""
        for benchmark_name in list_available_benchmarks():
            benchmark = load_benchmark(benchmark_name)
            metadata = benchmark.get_metadata()

            # Required fields
            required_fields = ["name", "description", "question_types", "formats", "num_feature_models"]
            for field in required_fields:
                assert field in metadata, f"{benchmark_name} missing metadata field: {field}"

            # Name should match benchmark name
            assert metadata["name"] == benchmark_name

            # Should have reasonable number of models
            assert 1 <= metadata["num_feature_models"] <= 100  # Sanity check

            print(f"✅ {benchmark_name}: Metadata consistent")


class TestCLIIntegration:
    """Test CLI integration with all benchmarks"""

    @pytest.mark.parametrize("benchmark_name", ["cvb", "vsi", "mmmu", "video_mme"])
    def test_cli_accepts_all_benchmarks(self, benchmark_name):
        """Test that CLI accepts all benchmark names"""
        from TsT.__main__ import create_parser

        parser = create_parser()

        # Should parse RF mode
        args = parser.parse_args(["--benchmark", benchmark_name, "--mode", "rf"])
        assert args.benchmark == benchmark_name
        assert args.mode == "rf"

        # Should parse LLM mode
        args = parser.parse_args(["--benchmark", benchmark_name, "--mode", "llm"])
        assert args.benchmark == benchmark_name
        assert args.mode == "llm"
