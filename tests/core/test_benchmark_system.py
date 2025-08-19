"""
Tests for benchmark registry and base classes.

These tests ensure that the benchmark system works correctly with proper
registration, loading, and interface compliance.
"""

import pytest
import pandas as pd

from TsT.core.benchmark import Benchmark, BenchmarkRegistry
from TsT.core.qa_models import SimpleBenchmarkQAModel
from TsT.core.protocols import QuestionAnswerBiasModel
from ..conftest import isolate_registry


class MockFeatureModel:
    """Mock feature-based model for testing"""

    def __init__(self, name="mock_feature", format="mc"):
        self.name = name
        self.format = format
        self.feature_cols = ["feat1", "feat2"]
        self.target_col_override = None

    def select_rows(self, df):
        return df

    def fit_feature_maps(self, train_df):
        pass

    def add_features(self, df):
        return df

    @property
    def task(self):
        return "clf" if self.format == "mc" else "reg"

    @property
    def metric(self):
        return "acc" if self.format == "mc" else "mra"


class TestBenchmarkRegistry:
    """Test benchmark registry functionality"""

    @isolate_registry
    def test_registry_starts_empty(self):
        """Test that registry starts with empty state for testing"""
        assert len(BenchmarkRegistry._benchmarks) == 0

    @isolate_registry
    def test_register_decorator(self):
        """Test that @BenchmarkRegistry.register works correctly"""

        @BenchmarkRegistry.register
        class TestBenchmark(Benchmark):
            name = "test_benchmark"
            description = "A test benchmark"

            def load_data(self):
                return pd.DataFrame({"test": [1, 2, 3]})

            def get_feature_based_models(self):
                return [MockFeatureModel("test_model")]

            def get_qa_model(self):
                return SimpleBenchmarkQAModel(benchmark_name=self.name, name=f"{self.name}_qa", format="mc")

        # Should be registered
        assert "test_benchmark" in BenchmarkRegistry._benchmarks
        assert BenchmarkRegistry._benchmarks["test_benchmark"] == TestBenchmark

    @isolate_registry
    def test_get_benchmark(self):
        """Test getting benchmark instances"""

        @BenchmarkRegistry.register
        class AnotherTestBenchmark(Benchmark):
            name = "another_test"

            def load_data(self):
                return pd.DataFrame()

            def get_feature_based_models(self):
                return []

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("another_test", "qa", "mc")

        # Should get instance
        benchmark = BenchmarkRegistry.get_benchmark("another_test")
        assert isinstance(benchmark, AnotherTestBenchmark)
        assert benchmark.name == "another_test"

    @isolate_registry
    def test_get_unknown_benchmark_raises_error(self):
        """Test that getting unknown benchmark raises appropriate error"""
        with pytest.raises(ValueError, match="Unknown benchmark: unknown"):
            BenchmarkRegistry.get_benchmark("unknown")

    @isolate_registry
    def test_list_benchmarks(self):
        """Test listing all registered benchmarks"""

        @BenchmarkRegistry.register
        class Benchmark1(Benchmark):
            name = "bench1"

            def load_data(self):
                return pd.DataFrame()

            def get_feature_based_models(self):
                return []

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("bench1", "qa", "mc")

        @BenchmarkRegistry.register
        class Benchmark2(Benchmark):
            name = "bench2"

            def load_data(self):
                return pd.DataFrame()

            def get_feature_based_models(self):
                return []

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("bench2", "qa", "mc")

        benchmarks = BenchmarkRegistry.list_benchmarks()
        assert "bench1" in benchmarks
        assert "bench2" in benchmarks

    @isolate_registry
    def test_get_all_benchmarks(self):
        """Test getting all benchmarks as instances"""

        @BenchmarkRegistry.register
        class AllBench1(Benchmark):
            name = "all1"

            def load_data(self):
                return pd.DataFrame()

            def get_feature_based_models(self):
                return []

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("all1", "qa", "mc")

        @BenchmarkRegistry.register
        class AllBench2(Benchmark):
            name = "all2"

            def load_data(self):
                return pd.DataFrame()

            def get_feature_based_models(self):
                return []

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("all2", "qa", "mc")

        all_benchmarks = BenchmarkRegistry.get_all_benchmarks()
        assert "all1" in all_benchmarks
        assert "all2" in all_benchmarks
        assert isinstance(all_benchmarks["all1"], AllBench1)
        assert isinstance(all_benchmarks["all2"], AllBench2)


class TestBenchmarkBaseClass:
    """Test Benchmark abstract base class"""

    def test_cannot_instantiate_abstract_base(self):
        """Test that Benchmark cannot be instantiated directly"""
        with pytest.raises(TypeError):
            Benchmark()

    def test_subclass_must_define_name(self):
        """Test that subclasses must define name attribute"""

        with pytest.raises(ValueError, match="must define a 'name' class attribute"):

            class BadBenchmark(Benchmark):
                # Missing name attribute
                def load_data(self):
                    return pd.DataFrame()

                def get_feature_based_models(self):
                    return []

                def get_qa_model(self):
                    return SimpleBenchmarkQAModel("test", "qa", "mc")

    def test_concrete_implementation_works(self):
        """Test that concrete implementation of Benchmark works"""

        class ConcreteBenchmark(Benchmark):
            name = "concrete_test"
            description = "A concrete test benchmark"

            def load_data(self):
                return pd.DataFrame({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})

            def get_feature_based_models(self):
                return [MockFeatureModel("model1", "mc"), MockFeatureModel("model2", "num")]

            def get_qa_model(self):
                return SimpleBenchmarkQAModel(benchmark_name=self.name, name=f"{self.name}_qa", format="mc")

        # Should be able to instantiate
        benchmark = ConcreteBenchmark()
        assert benchmark.name == "concrete_test"
        assert benchmark.description == "A concrete test benchmark"

        # Should be able to call methods
        data = benchmark.load_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2

        feature_models = benchmark.get_feature_based_models()
        assert len(feature_models) == 2
        assert all(hasattr(model, "feature_cols") for model in feature_models)

        qa_model = benchmark.get_qa_model()
        assert hasattr(qa_model, "benchmark_name")
        assert qa_model.benchmark_name == "concrete_test"

    def test_get_metadata_default_implementation(self):
        """Test default metadata generation"""

        class MetadataBenchmark(Benchmark):
            name = "metadata_test"
            description = "Testing metadata"

            def load_data(self):
                return pd.DataFrame()

            def get_feature_based_models(self):
                return [
                    MockFeatureModel("count_model", "mc"),
                    MockFeatureModel("size_model", "num"),
                    MockFeatureModel("relation_model", "mc"),
                ]

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("metadata_test", "qa", "oe")

        benchmark = MetadataBenchmark()
        metadata = benchmark.get_metadata()

        assert metadata["name"] == "metadata_test"
        assert metadata["description"] == "Testing metadata"
        assert metadata["num_feature_models"] == 3
        assert set(metadata["question_types"]) == {"count_model", "size_model", "relation_model"}

        # Check format mapping
        formats = metadata["formats"]
        assert "mc" in formats
        assert "num" in formats
        assert len(formats["mc"]) == 2  # count_model, relation_model
        assert len(formats["num"]) == 1  # size_model

    def test_format_mapping_helper(self):
        """Test the _get_format_mapping helper method"""

        class FormatBenchmark(Benchmark):
            name = "format_test"

            def load_data(self):
                return pd.DataFrame()

            def get_feature_based_models(self):
                return []

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("format_test", "qa", "mc")

        benchmark = FormatBenchmark()

        models = [
            MockFeatureModel("mc1", "mc"),
            MockFeatureModel("mc2", "mc"),
            MockFeatureModel("num1", "num"),
            MockFeatureModel("mc3", "mc"),
        ]

        format_mapping = benchmark._get_format_mapping(models)

        assert "mc" in format_mapping
        assert "num" in format_mapping
        assert len(format_mapping["mc"]) == 3
        assert len(format_mapping["num"]) == 1
        assert "mc1" in format_mapping["mc"]
        assert "mc2" in format_mapping["mc"]
        assert "mc3" in format_mapping["mc"]
        assert "num1" in format_mapping["num"]


class TestSimpleBenchmarkQAModel:
    """Test SimpleBenchmarkQAModel implementation"""

    def test_basic_creation(self):
        """Test basic QA model creation"""
        model = SimpleBenchmarkQAModel(benchmark_name="test_bench", name="test_qa", format="mc")

        assert model.benchmark_name == "test_bench"
        assert model.name == "test_qa"
        assert model.format == "mc"
        assert model.question_types is None
        assert model.target_col_override is None

    def test_select_rows_all_types(self):
        """Test select_rows when question_types is None (all types)"""
        model = SimpleBenchmarkQAModel("bench", "qa", "mc")

        df = pd.DataFrame({"question_type": ["type1", "type2", "type1", "type3"], "question": ["Q1", "Q2", "Q3", "Q4"]})

        selected = model.select_rows(df)
        assert len(selected) == 4  # Should select all rows

    def test_select_rows_filtered_types(self):
        """Test select_rows when question_types is specified"""
        model = SimpleBenchmarkQAModel(
            benchmark_name="bench", name="qa", format="mc", question_types=["type1", "type3"]
        )

        df = pd.DataFrame(
            {"question_type": ["type1", "type2", "type1", "type3", "type2"], "question": ["Q1", "Q2", "Q3", "Q4", "Q5"]}
        )

        selected = model.select_rows(df)
        assert len(selected) == 3  # Should select type1 and type3 only
        assert list(selected["question_type"]) == ["type1", "type1", "type3"]

    def test_prepare_instances_default(self):
        """Test default prepare_instances implementation"""
        model = SimpleBenchmarkQAModel("bench", "qa", "mc")

        df = pd.DataFrame({"test": [1, 2, 3]})
        prepared = model.prepare_instances(df)

        # Should return unchanged dataframe
        pd.testing.assert_frame_equal(prepared, df)

    def test_task_property(self):
        """Test task property for different formats"""
        mc_model = SimpleBenchmarkQAModel("bench", "qa", "mc")
        num_model = SimpleBenchmarkQAModel("bench", "qa", "num")
        oe_model = SimpleBenchmarkQAModel("bench", "qa", "oe")

        assert mc_model.task == "clf"
        assert num_model.task == "reg"
        assert oe_model.task == "oe"

    def test_metric_property(self):
        """Test metric property for different formats"""
        mc_model = SimpleBenchmarkQAModel("bench", "qa", "mc")
        num_model = SimpleBenchmarkQAModel("bench", "qa", "num")
        oe_model = SimpleBenchmarkQAModel("bench", "qa", "oe")

        assert mc_model.metric == "acc"
        assert num_model.metric == "mra"
        with pytest.raises(NotImplementedError):
            oe_model.metric

    def test_implements_qa_protocol(self):
        """Test that SimpleBenchmarkQAModel implements QuestionAnswerBiasModel protocol"""
        model = SimpleBenchmarkQAModel("test", "qa", "mc")

        # Should be usable as QuestionAnswerBiasModel
        def use_as_qa_model(qa_model: QuestionAnswerBiasModel):
            assert hasattr(qa_model, "benchmark_name")
            assert hasattr(qa_model, "name")
            assert hasattr(qa_model, "format")
            assert hasattr(qa_model, "target_col_override")
            assert hasattr(qa_model, "task")
            assert hasattr(qa_model, "metric")
            assert hasattr(qa_model, "select_rows")
            assert hasattr(qa_model, "prepare_instances")

        use_as_qa_model(model)


class TestBenchmarkIntegration:
    """Integration tests for the benchmark system"""

    @isolate_registry
    def test_complete_benchmark_workflow(self):
        """Test complete workflow from registration to usage"""

        @BenchmarkRegistry.register
        class WorkflowBenchmark(Benchmark):
            name = "workflow_test"
            description = "Complete workflow test"

            def load_data(self):
                return pd.DataFrame(
                    {
                        "question_type": ["count", "size", "count"],
                        "question": ["How many?", "How big?", "Count them"],
                        "answer": ["3", "large", "5"],
                    }
                )

            def get_feature_based_models(self):
                return [MockFeatureModel("count_model", "mc"), MockFeatureModel("size_model", "mc")]

            def get_qa_model(self):
                return SimpleBenchmarkQAModel(benchmark_name=self.name, name=f"{self.name}_qa", format="mc")

        # Test registration
        assert "workflow_test" in BenchmarkRegistry.list_benchmarks()

        # Test getting benchmark
        benchmark = BenchmarkRegistry.get_benchmark("workflow_test")
        assert isinstance(benchmark, WorkflowBenchmark)

        # Test benchmark functionality
        data = benchmark.load_data()
        assert len(data) == 3
        assert "question_type" in data.columns

        feature_models = benchmark.get_feature_based_models()
        assert len(feature_models) == 2

        qa_model = benchmark.get_qa_model()
        assert qa_model.benchmark_name == "workflow_test"

        # Test QA model functionality
        selected = qa_model.select_rows(data)
        assert len(selected) == 3  # Should select all

        metadata = benchmark.get_metadata()
        assert metadata["name"] == "workflow_test"
        assert len(metadata["question_types"]) == 2

    @isolate_registry
    def test_multiple_benchmarks_coexist(self):
        """Test that multiple benchmarks can be registered and work together"""

        @BenchmarkRegistry.register
        class Bench1(Benchmark):
            name = "bench1"

            def load_data(self):
                return pd.DataFrame({"test1": [1]})

            def get_feature_based_models(self):
                return [MockFeatureModel("model1")]

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("bench1", "qa1", "mc")

        @BenchmarkRegistry.register
        class Bench2(Benchmark):
            name = "bench2"

            def load_data(self):
                return pd.DataFrame({"test2": [2]})

            def get_feature_based_models(self):
                return [MockFeatureModel("model2")]

            def get_qa_model(self):
                return SimpleBenchmarkQAModel("bench2", "qa2", "num")

        # Both should be available
        benchmarks = BenchmarkRegistry.list_benchmarks()
        assert "bench1" in benchmarks
        assert "bench2" in benchmarks

        # Should be able to get both
        b1 = BenchmarkRegistry.get_benchmark("bench1")
        b2 = BenchmarkRegistry.get_benchmark("bench2")

        assert b1.name == "bench1"
        assert b2.name == "bench2"

        # Should have different data and models
        assert "test1" in b1.load_data().columns
        assert "test2" in b2.load_data().columns

        assert b1.get_qa_model().format == "mc"
        assert b2.get_qa_model().format == "num"
