"""
LLM evaluator for the unified evaluation framework.

This module contains the LLMEvaluator and related utilities for
training and evaluating LLM models on bias detection tasks.
"""

from typing import Optional, List, Literal
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from ezcolorlog import root_logger as logger

from ..llm.data.models import TestInstance, LLMPredictionResult
from ..llm.data.conversion import convert_to_blind_training_format, convert_to_blind_test_instances
from ..llm.predictors.vllm import VLLMPredictor
from ..llm.predictors.base import LLMPredictorInterface
from ..llm.trainers.llamafactory import create_llamafactory_trainer
from ..llm.trainable.predictor import TrainableLLMPredictor, TrainableLLMPredictorConfig
from ..llm.config import LLMRunConfig
from ...core.protocols import ModelEvaluator, BiasModel
from ...core.results import FoldResult, EvaluationResult
from ...utils import fuzzy_match, fuzzy_mra, parse_multi_choice_response


def score_llm(result: LLMPredictionResult, test_instance: TestInstance, format_type: Literal["mc", "num", "oe"]):
    match format_type:
        case "mc":
            pred_idx = parse_multi_choice_response(result.prediction, test_instance.options)
            gt_idx = parse_multi_choice_response(test_instance.ground_truth, test_instance.options)
            return float(pred_idx == gt_idx)
        case "num":
            return fuzzy_mra(result.prediction, test_instance.ground_truth)
        case "oe":
            return fuzzy_match(result.prediction, test_instance.ground_truth)
        case _:
            raise ValueError(f"Unknown format_type: {format_type}")


def evaluate_llm(
    predictor: LLMPredictorInterface,
    test_instances: List[TestInstance],
    format_type: Literal["mc", "num", "oe"],
) -> float:
    """
    Evaluate LLM on test data and return a score in [0, 1].

    - mc (multiple choice): per-instance scoring via parse_multi_choice_response using provided options
    - num (numeric): per-instance fuzzy MRA using robust numeric cleaning
    - oe (open-ended): per-instance fuzzy string matching
    """
    if not test_instances:
        return 0.0

    prediction_results = predictor.predict(test_instances)

    # Log first example for quick debugging
    if prediction_results and test_instances:
        logger.info(f"First instruction:\t{test_instances[0].instruction}")
        logger.info(f"First ground truth:\t{test_instances[0].ground_truth}")
        logger.info(f"  --> options:\t\t{test_instances[0].options}")
        logger.info(f"First prediction:\t\t{prediction_results[0].prediction}")
        logger.info(f"  --> Raw output:\t\t{prediction_results[0].raw_output}")

    # Compute per-instance scores then average
    scores = [score_llm(res, inst, format_type) for res, inst in zip(prediction_results, test_instances)]
    return float(np.mean(scores)) if scores else 0.0


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


class LLMEvaluator(ModelEvaluator):
    """LLM model evaluator for unified evaluation framework"""

    def __init__(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        llm_config: Optional[LLMRunConfig] = None,
    ):
        self.model = model
        self.df = df
        self.qdf = model.select_rows(df)
        self.target_col = target_col
        # Use provided run config or defaults
        self.llm_config = llm_config or LLMRunConfig()

        # Initialize LLM predictor
        self.llm_config_obj = self.llm_config.to_predictor_config()
        # Force-disable chat template for compatibility unless explicitly enabled in run config
        self.llm_config_obj.apply_chat_template = self.llm_config.apply_chat_template
        self.predictor = VLLMPredictor(self.llm_config_obj)

        # Initialize LlamaFactory trainer and composed trainable predictor
        self.trainer = create_llamafactory_trainer(**self.llm_config.to_trainer_config().__dict__)
        self.trainable = TrainableLLMPredictor(
            predictor=self.predictor,
            trainer=self.trainer,
            config=TrainableLLMPredictorConfig(reset_predictor_before_training=True),
        )

        # Get zero-shot baseline
        self.zero_shot_baseline = self.evaluate_zero_shot_baseline()

    def evaluate_zero_shot_baseline(self) -> float:
        """Evaluate zero-shot baseline performance."""
        logger.info(f"Evaluating zero-shot baseline for {self.model} with LLM predictor {self.predictor}")
        score = evaluate_llm_zero_shot(
            self.predictor,
            self.qdf,
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
            logger.info(f"Training adapter on fold #{fold_id} with {len(train_data)} samples")
            _adapter_info = self.trainable.train(train_data, temp_path)
            logger.info(f"Trained adapter on fold #{fold_id} with {len(train_data)} samples")

            # Evaluate on test fold using the same predictor with loaded adapter
            logger.info(f"Evaluating on fold #{fold_id} with {len(test_instances)} samples")
            fold_score = evaluate_llm(self.predictor, test_instances, self.model.format)
            improvement = fold_score - self.zero_shot_baseline
            logger.info(f"Fold #{fold_id} score: {fold_score:.2%}")
            logger.info(f"Fold #{fold_id} improvement: {improvement:.2%}")
            logger.info(f"Zero-shot baseline: {self.zero_shot_baseline:.2%}")

        return FoldResult(
            fold_id=fold_id,
            score=fold_score,
            train_size=len(train_df),
            test_size=len(test_df),
            metadata={
                "model_name": self.llm_config.model_name,
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

        evaluation_result.feature_importances = None  # TODO?
        evaluation_result.zero_shot_baseline = self.zero_shot_baseline
        evaluation_result.model_metadata.update(
            {
                "zero_shot_baseline": self.zero_shot_baseline,
                "improvement": improvement,
                "llm_config": self.llm_config,
                "total_samples": len(df),
            }
        )

        return evaluation_result
