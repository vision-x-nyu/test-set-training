"""
Tests for protocol compliance and type system.

These tests ensure that the new protocol system works correctly and that
existing models are compatible with the new BiasModel interface.
"""

import pytest
import pandas as pd
from typing import get_type_hints

from TsT.core.protocols import BiasModel, ModelEvaluator
from TsT.core.protocols import FeatureBasedBiasModel


class TestBiasModelProtocol:
    """Test BiasModel protocol definition and compliance"""

    def test_protocol_has_required_attributes(self):
        """Test that BiasModel protocol defines required attributes"""
        # Check that protocol has the expected attributes
        hints = get_type_hints(BiasModel)

        # These should be annotated in the protocol
        expected_attrs = ["name", "format", "target_col_override"]
        for attr in expected_attrs:
            assert attr in hints, f"BiasModel should define {attr}"

    def test_protocol_has_required_methods(self):
        """Test that BiasModel protocol defines required methods"""
        # Check that protocol has the expected methods
        protocol_methods = [name for name in dir(BiasModel) if not name.startswith("_")]

        expected_methods = ["select_rows", "task", "metric"]
        for method in expected_methods:
            assert method in protocol_methods, f"BiasModel should define {method}"


class TestFeatureBasedBiasModelInheritance:
    """Test that FeatureBasedBiasModel correctly inherits from BiasModel"""

    def test_qtype_implements_bias_model(self):
        """Test that FeatureBasedBiasModel implements BiasModel protocol"""
        # Note: Cannot use issubclass() with Protocols that have non-method members
        # Instead test that FeatureBasedBiasModel has all BiasModel functionality
        from typing import get_type_hints

        bias_model_hints = get_type_hints(BiasModel)
        qtype_hints = get_type_hints(FeatureBasedBiasModel)

        # FeatureBasedBiasModel should have all BiasModel attributes
        for attr in bias_model_hints:
            assert attr in qtype_hints, f"FeatureBasedBiasModel should have {attr} attribute"

    def test_qtype_has_bias_model_methods(self):
        """Test that FeatureBasedBiasModel has all BiasModel methods"""
        # FeatureBasedBiasModel should have all BiasModel methods
        bias_model_methods = [name for name in dir(BiasModel) if not name.startswith("_")]
        qtype_methods = [name for name in dir(FeatureBasedBiasModel) if not name.startswith("_")]

        for method in bias_model_methods:
            assert method in qtype_methods, f"FeatureBasedBiasModel should have {method} method"

    def test_qtype_has_additional_methods(self):
        """Test that FeatureBasedBiasModel has its additional feature-based methods"""
        qtype_methods = [name for name in dir(FeatureBasedBiasModel) if not name.startswith("_")]

        # FeatureBasedBiasModel should have these additional methods for feature engineering
        # Note: feature_cols is an attribute annotation, not a method
        additional_methods = ["fit_feature_maps", "add_features"]
        for method in additional_methods:
            assert method in qtype_methods, f"FeatureBasedBiasModel should have {method}"

        # Check that feature_cols is annotated
        from typing import get_type_hints

        hints = get_type_hints(FeatureBasedBiasModel)
        assert "feature_cols" in hints, "FeatureBasedBiasModel should have feature_cols annotation"


class MockModel:
    """Mock model that implements BiasModel protocol"""

    def __init__(self, name="mock", format="mc", task="clf", metric="acc"):
        self.name = name
        self.format = format
        self._task = task
        self._metric = metric
        self.target_col_override = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @property
    def task(self):
        return self._task

    @property
    def metric(self):
        return self._metric


class TestProtocolCompliance:
    """Test that models can be used as BiasModel protocol"""

    def test_mock_model_is_bias_model(self):
        """Test that mock model satisfies BiasModel protocol"""
        model = MockModel()

        # Should be able to use as BiasModel
        def use_bias_model(bias_model: BiasModel):
            # Access protocol attributes
            assert hasattr(bias_model, "name")
            assert hasattr(bias_model, "format")
            assert hasattr(bias_model, "target_col_override")

            # Call protocol methods
            df = pd.DataFrame({"col1": [1, 2, 3]})
            result = bias_model.select_rows(df)
            assert isinstance(result, pd.DataFrame)

            task = bias_model.task
            assert task in ["clf", "reg"]

            metric = bias_model.metric
            assert metric in ["acc", "mra"]

        # Should not raise any errors
        use_bias_model(model)

    def test_protocol_duck_typing(self):
        """Test that any object with the right interface works"""

        class DuckTypedModel:
            """Model that implements interface without explicit inheritance"""

            name = "duck_model"
            format = "num"
            target_col_override = None

            def select_rows(self, df):
                return df

            @property
            def task(self):
                return "reg"

            @property
            def metric(self):
                return "mra"

        model = DuckTypedModel()

        # Should work with BiasModel interface
        def test_interface(bias_model: BiasModel):
            assert bias_model.name == "duck_model"
            assert bias_model.format == "num"
            assert bias_model.task == "reg"
            assert bias_model.metric == "mra"

        test_interface(model)


class TestRealModelCompliance:
    """Test that real benchmark models comply with protocols"""

    def test_video_mme_model_compliance(self):
        """Test that Video-MME models work as BiasModel"""
        try:
            from TsT.benchmarks.video_mme import get_models

            models = get_models()
            if not models:
                pytest.skip("No Video-MME models available")

            model = models[0]

            # Test BiasModel interface
            def check_bias_model_interface(bias_model: BiasModel):
                # Required attributes
                assert hasattr(bias_model, "name")
                assert hasattr(bias_model, "format")
                assert hasattr(bias_model, "target_col_override")

                # Required methods
                assert hasattr(bias_model, "select_rows")
                assert hasattr(bias_model, "task")
                assert hasattr(bias_model, "metric")

                # Test method calls with proper Video-MME columns
                df = pd.DataFrame(
                    {
                        "test": [1, 2, 3],
                        "duration": [1.0, 2.0, 3.0],
                        "domain": ["test"] * 3,
                        "sub_category": ["test"] * 3,
                        "task_type": ["test"] * 3,
                        "answer": ["A", "B", "C"],
                    }
                )
                result = bias_model.select_rows(df)
                assert isinstance(result, pd.DataFrame)

                task = bias_model.task
                assert task in ["clf", "reg"]

                metric = bias_model.metric
                assert metric in ["acc", "mra"]

            # Should work without errors
            check_bias_model_interface(model)

        except ImportError:
            pytest.skip("Video-MME benchmark not available")

    def test_video_mme_qtype_compliance(self):
        """Test that Video-MME models still work as FeatureBasedBiasModel"""
        try:
            from TsT.benchmarks.video_mme import get_models

            models = get_models()
            if not models:
                pytest.skip("No Video-MME models available")

            model = models[0]

            # Test FeatureBasedBiasModel interface (should still work)
            def check_qtype_interface(qtype_model: FeatureBasedBiasModel):
                # BiasModel attributes
                assert hasattr(qtype_model, "name")
                assert hasattr(qtype_model, "format")

                # FeatureBasedBiasModel-specific attributes
                assert hasattr(qtype_model, "feature_cols")

                # Methods
                assert hasattr(qtype_model, "fit_feature_maps")
                assert hasattr(qtype_model, "add_features")
                assert hasattr(qtype_model, "select_rows")

                # Test properties
                task = qtype_model.task
                assert task in ["clf", "reg"]

                metric = qtype_model.metric
                assert metric in ["acc", "mra"]

            # Should work without errors
            check_qtype_interface(model)

        except ImportError:
            pytest.skip("Video-MME benchmark not available")


class TestModelEvaluatorProtocol:
    """Test ModelEvaluator abstract base class"""

    def test_cannot_instantiate_abstract_base(self):
        """Test that ModelEvaluator cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ModelEvaluator()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation of ModelEvaluator works"""

        class ConcreteEvaluator(ModelEvaluator):
            def train_and_evaluate_fold(self, model, train_df, test_df, target_col, fold_num, seed):
                return 0.8

        # Should be able to instantiate concrete implementation
        evaluator = ConcreteEvaluator()
        assert evaluator is not None

        # Should be able to call the method
        result = evaluator.train_and_evaluate_fold(
            model=MockModel(), train_df=pd.DataFrame(), test_df=pd.DataFrame(), target_col="test", fold_num=1, seed=42
        )
        assert result == 0.8

    def test_missing_implementation_fails(self):
        """Test that missing evaluate_fold implementation fails"""

        class IncompleteEvaluator(ModelEvaluator):
            pass  # Missing evaluate_fold implementation

        # Should not be able to instantiate
        with pytest.raises(TypeError):
            IncompleteEvaluator()


class TestTypeAnnotations:
    """Test that type annotations work correctly"""

    def test_bias_model_type_hints(self):
        """Test BiasModel type hints"""
        hints = get_type_hints(BiasModel)

        # Check specific type annotations
        assert "name" in hints
        assert "format" in hints
        assert "target_col_override" in hints

    def test_model_evaluator_type_hints(self):
        """Test ModelEvaluator type hints"""
        from inspect import signature

        # Check train_and_evaluate_fold method signature
        sig = signature(ModelEvaluator.train_and_evaluate_fold)
        params = sig.parameters

        assert "model" in params
        assert "train_df" in params
        assert "test_df" in params
        assert "target_col" in params
        assert "fold_id" in params  # Changed from fold_num to fold_id
        assert "seed" in params
