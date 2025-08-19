"""
Tests for QuestionAnswerBiasModel protocol.

These tests ensure that the new QA protocol works correctly and that
models can implement it properly.
"""

import pandas as pd
from typing import get_type_hints

from TsT.core.protocols import QuestionAnswerBiasModel, BiasModel


class TestQuestionAnswerBiasModelProtocol:
    """Test QuestionAnswerBiasModel protocol definition and compliance"""

    def test_protocol_has_required_attributes(self):
        """Test that QuestionAnswerBiasModel protocol defines required attributes"""
        # Check that protocol has the expected attributes
        hints = get_type_hints(QuestionAnswerBiasModel)

        # Should have all BiasModel attributes plus benchmark_name
        expected_attrs = ["name", "format", "target_col_override", "benchmark_name"]
        for attr in expected_attrs:
            assert attr in hints, f"QuestionAnswerBiasModel should define {attr}"

    def test_protocol_has_required_methods(self):
        """Test that QuestionAnswerBiasModel protocol defines required methods"""
        # Check that protocol has the expected methods
        protocol_methods = [name for name in dir(QuestionAnswerBiasModel) if not name.startswith("_")]

        # Should have all BiasModel methods plus prepare_instances
        expected_methods = ["select_rows", "task", "metric", "prepare_instances"]
        for method in expected_methods:
            assert method in protocol_methods, f"QuestionAnswerBiasModel should define {method}"

    def test_protocol_inherits_from_bias_model(self):
        """Test that QuestionAnswerBiasModel properly inherits from BiasModel"""
        # Check that QA model has all BiasModel attributes
        bias_model_hints = get_type_hints(BiasModel)
        qa_model_hints = get_type_hints(QuestionAnswerBiasModel)

        for attr in bias_model_hints:
            assert attr in qa_model_hints, f"QuestionAnswerBiasModel should inherit {attr} from BiasModel"


class MockQAModel:
    """Mock model that implements QuestionAnswerBiasModel protocol"""

    def __init__(self, name="mock_qa", benchmark="test", format="mc", task="clf", metric="acc"):
        self.name = name
        self.benchmark_name = benchmark
        self.format = format
        self._task = task
        self._metric = metric
        self.target_col_override = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def prepare_instances(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add some mock preprocessing
        df_copy = df.copy()
        df_copy["processed"] = True
        return df_copy

    @property
    def task(self):
        return self._task

    @property
    def metric(self):
        return self._metric


class TestQuestionAnswerBiasModelCompliance:
    """Test that models can properly implement QuestionAnswerBiasModel protocol"""

    def test_mock_qa_model_is_qa_bias_model(self):
        """Test that mock model satisfies QuestionAnswerBiasModel protocol"""
        model = MockQAModel()

        # Should be able to use as QuestionAnswerBiasModel
        def use_qa_bias_model(qa_model: QuestionAnswerBiasModel):
            # Access protocol attributes
            assert hasattr(qa_model, "name")
            assert hasattr(qa_model, "format")
            assert hasattr(qa_model, "target_col_override")
            assert hasattr(qa_model, "benchmark_name")

            # Call protocol methods
            df = pd.DataFrame({"col1": [1, 2, 3]})

            result = qa_model.select_rows(df)
            assert isinstance(result, pd.DataFrame)

            processed = qa_model.prepare_instances(df)
            assert isinstance(processed, pd.DataFrame)
            assert "processed" in processed.columns

            task = qa_model.task
            assert task in ["clf", "reg"]

            metric = qa_model.metric
            assert metric in ["acc", "mra"]

        # Should not raise any errors
        use_qa_bias_model(model)

    def test_qa_model_attributes(self):
        """Test that QA model has correct attributes"""
        model = MockQAModel(name="test_qa", benchmark="cvb", format="num")

        assert model.name == "test_qa"
        assert model.benchmark_name == "cvb"
        assert model.format == "num"
        assert model.target_col_override is None

    def test_qa_model_methods(self):
        """Test that QA model methods work correctly"""
        model = MockQAModel()

        # Test select_rows
        df = pd.DataFrame({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})
        selected = model.select_rows(df)
        assert len(selected) == 2
        assert list(selected.columns) == ["question", "answer"]

        # Test prepare_instances
        processed = model.prepare_instances(df)
        assert len(processed) == 2
        assert "processed" in processed.columns
        assert all(processed["processed"])

    def test_qa_model_properties(self):
        """Test that QA model properties work correctly"""
        clf_model = MockQAModel(format="mc", task="clf", metric="acc")
        reg_model = MockQAModel(format="num", task="reg", metric="mra")

        assert clf_model.task == "clf"
        assert clf_model.metric == "acc"

        assert reg_model.task == "reg"
        assert reg_model.metric == "mra"

    def test_qa_model_duck_typing(self):
        """Test that any object with the right interface works as QA model"""

        class DuckTypedQAModel:
            """Model that implements QA interface without explicit inheritance"""

            name = "duck_qa_model"
            benchmark_name = "test_benchmark"
            format = "mc"
            target_col_override = None

            def select_rows(self, df):
                return df.copy()

            def prepare_instances(self, df):
                df_copy = df.copy()
                df_copy["duck_processed"] = True
                return df_copy

            @property
            def task(self):
                return "clf"

            @property
            def metric(self):
                return "acc"

        model = DuckTypedQAModel()

        # Should work with QuestionAnswerBiasModel interface
        def test_interface(qa_model: QuestionAnswerBiasModel):
            assert qa_model.name == "duck_qa_model"
            assert qa_model.benchmark_name == "test_benchmark"
            assert qa_model.format == "mc"
            assert qa_model.task == "clf"
            assert qa_model.metric == "acc"

            df = pd.DataFrame({"test": [1, 2, 3]})
            processed = qa_model.prepare_instances(df)
            assert "duck_processed" in processed.columns

        test_interface(model)


class TestMixedModelTypes:
    """Test interaction between different model types"""

    def test_both_model_types_coexist(self):
        """Test that both FeatureBasedBiasModel and QuestionAnswerBiasModel can be used together"""

        # Mock feature-based model
        class MockFeatureModel:
            name = "mock_feature"
            format = "mc"
            feature_cols = ["feat1", "feat2"]
            target_col_override = None

            def select_rows(self, df):
                return df

            def fit_feature_maps(self, train_df):
                pass

            def add_features(self, df):
                df_copy = df.copy()
                df_copy["feat1"] = 1.0
                df_copy["feat2"] = 2.0
                return df_copy

            @property
            def task(self):
                return "clf"

            @property
            def metric(self):
                return "acc"

        qa_model = MockQAModel()
        feature_model = MockFeatureModel()

        # Both should work as BiasModel
        def use_as_bias_model(model: BiasModel):
            assert hasattr(model, "name")
            assert hasattr(model, "format")
            assert hasattr(model, "task")
            assert hasattr(model, "metric")

        use_as_bias_model(qa_model)
        use_as_bias_model(feature_model)

        # But they have different specific capabilities
        assert hasattr(qa_model, "benchmark_name")
        assert hasattr(qa_model, "prepare_instances")

        assert hasattr(feature_model, "feature_cols")
        assert hasattr(feature_model, "fit_feature_maps")
        assert hasattr(feature_model, "add_features")
