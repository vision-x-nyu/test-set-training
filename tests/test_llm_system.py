"""
Integration tests for the TsT LLM system.

This module tests the complete LLM infrastructure including predictors, trainers,
data models, and integration with the evaluation framework.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from TsT.llm import (
    create_vllm_predictor,
    create_llamafactory_trainer,
    create_trainable_predictor,
    TrainingDatum,
    TestInstance,
    LLMPredictionResult,
)
from TsT.core.evaluators import LLMEvaluator


class TestLLMComponents:
    """Test basic functionality of LLM components"""

    def test_data_models(self):
        """Test Pydantic data models work correctly"""
        # Test training datum
        training_datum = TrainingDatum(instruction="What is 2 + 2?", response="4", metadata={"test": True})
        assert training_datum.instruction == "What is 2 + 2?"
        assert training_datum.response == "4"
        assert training_datum.metadata is not None
        assert training_datum.metadata["test"] is True

        # Test test instance
        test_instance = TestInstance(instruction="What is 3 + 3?", instance_id="test_1", ground_truth="6")
        assert test_instance.instruction == "What is 3 + 3?"
        assert test_instance.instance_id == "test_1"
        assert test_instance.ground_truth == "6"

        # Test prediction result
        prediction = LLMPredictionResult(instance_id="test_1", prediction="6", confidence=0.95)
        assert prediction.instance_id == "test_1"
        assert prediction.prediction == "6"
        assert prediction.confidence == 0.95

    @patch("TsT.llm.predictors.vllm.LLM")
    @patch("TsT.llm.predictors.vllm.AutoTokenizer")
    def test_vllm_predictor_creation(self, mock_tokenizer, mock_llm):
        """Test that vLLM predictor can be created"""
        # Mock the tokenizer and LLM
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_llm.return_value = Mock()

        # Test predictor creation
        predictor = create_vllm_predictor(model_name="google/gemma-2-2b-it", max_seq_length=256)

        assert predictor is not None
        assert predictor.config.model_name == "google/gemma-2-2b-it"
        assert predictor.config.max_seq_length == 256

    def test_llamafactory_trainer_creation(self):
        """Test that LlamaFactory trainer can be created"""
        trainer = create_llamafactory_trainer(model_name="google/gemma-2-2b-it", learning_rate=1e-4, num_epochs=1)

        assert trainer is not None
        assert trainer.config.model_name == "google/gemma-2-2b-it"
        assert trainer.config.learning_rate == 1e-4
        assert trainer.config.num_epochs == 1

    @patch("TsT.llm.predictors.vllm.LLM")
    @patch("TsT.llm.predictors.vllm.AutoTokenizer")
    def test_trainable_predictor_creation(self, mock_tokenizer, mock_llm):
        """Test that trainable predictor can be created"""
        # Mock the LLM components
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_llm.return_value = Mock()

        # Create components
        predictor = create_vllm_predictor()
        trainer = create_llamafactory_trainer()
        trainable = create_trainable_predictor(predictor, trainer)

        assert trainable is not None
        assert trainable.predictor is predictor
        assert trainable.trainer is trainer


class TestLLMDataConversion:
    """Test LLM data conversion utilities"""

    def test_benchmark_data_conversion(self):
        """Test conversion from benchmark data to LLM format"""
        from TsT.llm.data.conversion import convert_to_tst_training_format

        # Create sample benchmark data
        df = pd.DataFrame(
            {
                "question": ["What is the color?", "How many objects?"],
                "gt_idx": ["A", "B"],
                "options": [["A: Red", "B: Blue"], ["A: 1", "B: 2"]],
            }
        )

        # Convert to training format
        training_data = convert_to_tst_training_format(df=df, target_col="gt_idx", format_type="mc")

        assert len(training_data) == 2
        assert training_data[0].instruction == "Answer the following question: What is the color?"
        assert training_data[0].response == "The answer is A."
        assert training_data[1].instruction == "Answer the following question: How many objects?"
        assert training_data[1].response == "The answer is B."


@pytest.mark.skip(reason="Requires GPU and model downloads")
class TestLLMFullTrainingPipeline:
    """Full end-to-end training pipeline tests that require GPU resources"""

    def test_end_to_end_training_and_inference(self):
        """Test complete training and inference pipeline"""
        # This would test the full pipeline but requires:
        # 1. GPU access
        # 2. Model downloads (several GB)
        # 3. LlamaFactory installation
        # 4. Significant computation time

        # Create sample data
        training_data = [
            TrainingDatum(instruction="What is 1+1?", response="2"),
            TrainingDatum(instruction="What is 2+2?", response="4"),
        ]

        test_instances = [TestInstance(instruction="What is 3+3?", instance_id="test_1", ground_truth="6")]

        # Create trainable predictor
        predictor = create_vllm_predictor()
        trainer = create_llamafactory_trainer(num_epochs=1)
        trainable = create_trainable_predictor(predictor, trainer)

        # Train and predict
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_info = trainable.train(training_data, Path(temp_dir))
            predictions = trainable.predict(test_instances)

            assert adapter_info is not None
            assert len(predictions) == 1
            assert predictions[0].instance_id == "test_1"


class TestLLMTrainingWithMocks:
    """Test LLM training pipeline with mocked dependencies"""

    @patch("pathlib.Path.exists")
    @patch("TsT.llm.trainers.llamafactory.run_llama_factory_training")
    @patch("TsT.llm.predictors.vllm.LLM")
    @patch("TsT.llm.predictors.vllm.AutoTokenizer")
    def test_llamafactory_training_with_mocks(self, mock_tokenizer, mock_llm, mock_training, mock_path_exists):
        """Test LlamaFactory training pipeline with mocked dependencies"""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_training.return_value = None
        mock_path_exists.return_value = True  # Pretend adapter path exists

        # Create components
        predictor = create_vllm_predictor()
        trainer = create_llamafactory_trainer(num_epochs=1)

        # Create a trainable predictor that doesn't reset the predictor before training for this test
        from TsT.llm.trainable.predictor import TrainableLLMPredictorConfig

        config = TrainableLLMPredictorConfig(reset_predictor_before_training=False)
        trainable = create_trainable_predictor(predictor, trainer)
        trainable.config = config

        # Create sample training data (need at least 2 examples)
        training_data = [
            TrainingDatum(instruction="Test question 1", response="Test answer 1"),
            TrainingDatum(instruction="Test question 2", response="Test answer 2"),
        ]

        # Test training
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_info = trainable.train(training_data, Path(temp_dir))

            # Verify training was called
            mock_training.assert_called_once()
            assert adapter_info.training_size == 2
            assert adapter_info.model_name == "google/gemma-2-2b-it"

    def test_llm_fold_evaluator(self):
        """Test that LLM fold evaluator works for LLM evaluation"""
        # Create mock model and data for LLMEvaluator constructor
        from unittest.mock import Mock

        mock_model = Mock()
        mock_model.name = "test_model"
        mock_model.format = "mc"

        test_df = pd.DataFrame({"question": ["What is 2+2?", "What is 3+3?"], "gt_idx": [0, 1]})

        # Mock the VLLMPredictor and evaluate_llm_zero_shot to avoid actual LLM calls
        with patch("TsT.core.evaluators.VLLMPredictor"):
            with patch("TsT.core.evaluators.evaluate_llm_zero_shot") as mock_eval:
                mock_eval.return_value = 0.5

                evaluator = LLMEvaluator(
                    model=mock_model,
                    df=test_df,
                    target_col="gt_idx",
                    llm_config={
                        "model_name": "test/model",
                        "batch_size": 4,
                        "max_seq_length": 512,
                        "learning_rate": 2e-4,
                        "num_epochs": 1,
                    },
                )

                assert evaluator is not None
                # The LLM evaluator should have the required method
                assert hasattr(evaluator, "train_and_evaluate_fold")


if __name__ == "__main__":
    pytest.main([__file__])
