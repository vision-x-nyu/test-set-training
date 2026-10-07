"""
Tests for the benchmark registry and the Benchmark base class.
"""

import pandas as pd
import pytest

from TsT.core.benchmark import BUILTIN_BENCHMARKS, Benchmark, BenchmarkRegistry
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


def _make_benchmark(bench_name, models=()):
    class _Bench(Benchmark):
        name = bench_name

        def load_data(self, revision=None):
            return pd.DataFrame({"revision": [revision]})

        def get_feature_based_models(self, feature_set="default"):
            self.check_feature_set(feature_set)
            return list(models)

    return _Bench


class TestBenchmarkRegistry:
    @isolate_registry
    def test_registry_starts_empty(self):
        assert len(BenchmarkRegistry._benchmarks) == 0

    @isolate_registry
    def test_register_and_get(self):
        bench_cls = BenchmarkRegistry.register(_make_benchmark("test_benchmark"))
        assert BenchmarkRegistry._benchmarks["test_benchmark"] is bench_cls
        instance = BenchmarkRegistry.get_benchmark("test_benchmark")
        assert isinstance(instance, bench_cls)
        assert instance is not BenchmarkRegistry.get_benchmark("test_benchmark")

    @isolate_registry
    def test_get_unknown_benchmark_raises_error(self):
        with pytest.raises(ValueError, match="Unknown benchmark: unknown"):
            BenchmarkRegistry.get_benchmark("unknown")

    @isolate_registry
    def test_list_and_get_all(self):
        BenchmarkRegistry.register(_make_benchmark("bench1"))
        BenchmarkRegistry.register(_make_benchmark("bench2"))
        assert set(BenchmarkRegistry.list_benchmarks()) >= {"bench1", "bench2"}
        assert set(BenchmarkRegistry.get_all_benchmarks()) >= {"bench1", "bench2"}

    def test_builtin_benchmarks_are_registered(self):
        assert set(BUILTIN_BENCHMARKS) == {"vsi", "cvb", "mmmu", "video_mme", "mmstar"}
        assert set(BenchmarkRegistry.list_benchmarks()) >= set(BUILTIN_BENCHMARKS)


class TestBenchmarkBaseClass:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Benchmark()

    def test_subclass_must_define_name(self):
        with pytest.raises(ValueError, match="must define a 'name' class attribute"):

            class BadBenchmark(Benchmark):
                def load_data(self, revision=None):
                    return pd.DataFrame()

                def get_feature_based_models(self, feature_set="default"):
                    return []

    def test_minimal_subclass_needs_only_data_and_feature_models(self):
        bench = _make_benchmark("minimal", [MockFeatureModel("m1", "mc"), MockFeatureModel("m2", "num")])()
        assert bench.load_data(revision="abc")["revision"].tolist() == ["abc"]
        assert [m.name for m in bench.get_feature_based_models()] == ["m1", "m2"]
        assert bench.feature_sets == ("default",) and bench.id_col == "id"
        assert bench.hf_repo is None and bench.default_revision is None

    def test_feature_set_check(self):
        bench = _make_benchmark("fs")()
        bench.check_feature_set("default")
        with pytest.raises(ValueError, match="Unknown feature set 'paper'"):
            bench.get_feature_based_models("paper")

    def test_llm_and_ibp_hooks_are_stubs(self):
        bench = _make_benchmark("stubs")()
        with pytest.raises(NotImplementedError, match="TsT-LLM"):
            bench.get_qa_models()
        with pytest.raises(NotImplementedError, match="IBP"):
            bench.get_ibp_strategy()

    def test_get_metadata_default_implementation(self):
        models = [MockFeatureModel("count", "mc"), MockFeatureModel("size", "num"), MockFeatureModel("rel", "mc")]
        bench = _make_benchmark("metadata_test", models)()
        metadata = bench.get_metadata()
        assert metadata["name"] == "metadata_test"
        assert metadata["num_feature_models"] == 3
        assert metadata["question_types"] == ["count", "size", "rel"]
        assert metadata["formats"] == {"mc": ["count", "rel"], "num": ["size"]}
        assert metadata["feature_sets"] == ["default"]
