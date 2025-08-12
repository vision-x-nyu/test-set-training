"""
LLM evaluator for the unified evaluation framework.

This module contains the LLMEvaluator and related utilities for
training and evaluating LLM models on bias detection tasks.
"""

from typing import Dict, Any, Optional, List, Literal
import tempfile
from pathlib import Path

import pandas as pd
from ezcolorlog import root_logger as logger

from ...core.protocols import ModelEvaluator, BiasModel
from ...core.results import FoldResult, EvaluationResult
from ..llm.data.models import TestInstance
from ..llm.data.conversion import convert_to_blind_training_format, convert_to_blind_test_instances
from ..llm.predictors.vllm import VLLMPredictor, VLLMPredictorConfig
from ..llm.predictors.base import LLMPredictorInterface
from ..llm.trainers.llamafactory import create_llamafactory_trainer
from ..llm.trainable.predictor import TrainableLLMPredictor, TrainableLLMPredictorConfig


def score_llm():
    raise NotImplementedError("LLM scoring not implemented")


def evaluate_llm_zero_shot(
    predictor: LLMPredictorInterface,
    df: pd.DataFrame,
    target_col: str,
    format_type: Literal["mc", "num", "oe"],
) -> float:
    """
    Evaluate zero-shot baseline performance.
    """
    logger.info(f"Evaluating zero-shot baseline for {format_type} format")

    # Convert to TestInstance objects directly
    test_instances = convert_to_blind_test_instances(
        df=df,
        target_col=target_col,
        format_type=format_type,
        instruction_template="Answer the following question: {question}",
        id_prefix="zero_shot",
    )

    return evaluate_llm(predictor, test_instances, format_type)


def evaluate_llm(
    predictor: LLMPredictorInterface,
    test_instances: List[TestInstance],
    format_type: Literal["mc", "num", "oe"],
) -> float:
    """
    Evaluate LLM on test data and return accuracy.
    """
    # No need to convert - already TestInstance objects!
    prediction_results = predictor.predict(test_instances)
    predictions = [result.prediction for result in prediction_results]
    ground_truth = [instance.ground_truth for instance in test_instances]

    # print the first prediction and ground truth
    if predictions and ground_truth:
        logger.info(f"First prediction: {predictions[0]}")
        logger.info(f"First ground truth: {ground_truth[0]}")

    # TODO: make an evaluation function that takes LLMPredictionResult objects and returns a score
    # HACK [temporary]: manually score here

    # Calculate accuracy
    if format_type == "mc":
        # For multiple choice, exact match
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip().upper() == gt.strip().upper())
    else:
        # For numerical, use relative accuracy or exact match
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip() == gt.strip())

    return correct / len(test_instances) if test_instances else 0.0


class LLMEvaluator(ModelEvaluator):
    """LLM model evaluator for unified evaluation framework"""

    default_llm_config = {
        "model_name": "google/gemma-2-2b-it",
        "batch_size": 32,
        "learning_rate": 2e-4,
        "num_epochs": 5,
        "lora_rank": 8,
        "lora_alpha": 16,
        "max_seq_length": 512,
    }

    def __init__(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.df = df
        self.target_col = target_col
        if llm_config is None:
            logger.warning(f"No LLM config provided, using default config: {self.default_llm_config}")
            llm_config = self.default_llm_config
        self.llm_config = llm_config

        # Initialize LLM predictor
        self.llm_config_obj = VLLMPredictorConfig(
            model_name=self.llm_config["model_name"],
            batch_size=self.llm_config["batch_size"],
            max_seq_length=self.llm_config["max_seq_length"],
            apply_chat_template=False,  # Disable for compatibility with Gemma and other models
        )
        self.predictor = VLLMPredictor(self.llm_config_obj)

        # Initialize LlamaFactory trainer and composed trainable predictor
        self.trainer = create_llamafactory_trainer(
            model_name=self.llm_config["model_name"],
            learning_rate=self.llm_config["learning_rate"],
            num_epochs=self.llm_config["num_epochs"],
            batch_size=self.llm_config["batch_size"],
            lora_rank=self.llm_config.get("lora_rank", 8),
            max_seq_length=self.llm_config["max_seq_length"],
            template=self.llm_config.get("template", "gemma"),
        )
        self.trainable = TrainableLLMPredictor(
            predictor=self.predictor,
            trainer=self.trainer,
            config=TrainableLLMPredictorConfig(reset_predictor_before_training=True),
        )

        self.zero_shot_baseline = self.evaluate_zero_shot_baseline()

    def evaluate_zero_shot_baseline(self) -> float:
        """Evaluate zero-shot baseline performance."""
        logger.info(f"Evaluating zero-shot baseline for {self.model} with LLM predictor {self.predictor}")
        score = evaluate_llm_zero_shot(
            self.predictor,
            self.df,
            self.target_col,
            self.model.format,
        )
        self.predictor.reset()
        logger.info(f"Zero-shot baseline score: {score:.2%}")
        return score

    def train_and_evaluate_fold(
        self,
        model,  # BiasModel
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Train + Evaluate LLM on a single fold"""

        # Create training dataset using Pydantic models
        train_data = convert_to_blind_training_format(train_df, target_col, self.model.format)

        # Create test instances using Pydantic models
        test_instances = convert_to_blind_test_instances(
            df=test_df,
            target_col=target_col,
            format_type=self.model.format,
            instruction_template="Answer the following question: {question}",
            id_prefix=f"fold_{fold_id}",
        )

        # Fine-tune LLM on training fold
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Train adapter on the training fold (loads adapter into predictor)
            _adapter_info = self.trainable.train(train_data, temp_path)

            # Evaluate on test fold using the same predictor with loaded adapter
            fold_score = evaluate_llm(self.predictor, test_instances, self.model.format)

        return FoldResult(
            fold_id=fold_id,
            score=fold_score,
            train_size=len(train_df),
            test_size=len(test_df),
            metadata={
                "model_name": self.llm_config.get("model_name", "unknown"),
                "llm_config": self.llm_config,
            },
        )

    def process_results(
        self,
        model,  # BiasModel
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Add LLM-specific metadata"""
        # Calculate improvement over zero-shot
        if self.zero_shot_baseline is None:
            raise NotImplementedError(
                "Zero-shot baseline is required for LLM post-processing. "
                "This should be provided when creating LLMPostProcessor."
            )

        improvement = evaluation_result.overall_mean - self.zero_shot_baseline

        # Mock feature importances for compatibility
        feature_importances = pd.DataFrame(
            {
                "feature": ["llm_finetuning", "zero_shot_baseline", "improvement"],
                "importance": [evaluation_result.overall_mean, self.zero_shot_baseline, improvement],
            }
        )

        evaluation_result.feature_importances = feature_importances
        evaluation_result.model_metadata.update(
            {
                "zero_shot_baseline": self.zero_shot_baseline,
                "improvement": improvement,
                "llm_config": self.llm_config,
                "total_samples": len(df),
            }
        )

        return evaluation_result
