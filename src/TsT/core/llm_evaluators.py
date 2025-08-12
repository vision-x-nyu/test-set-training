"""
LLM evaluators for the unified evaluation framework.

This module extends the core evaluation system to support LLM-based bias detection,
integrating with the new LLM infrastructure while maintaining compatibility with
the existing evaluation pipeline.
"""

import tempfile
from pathlib import Path
from typing import Literal
import pandas as pd

from .protocols import ModelEvaluator, BiasModel
from ..llm.trainable.predictor import TrainableLLMPredictor
from ..llm.data.conversion import convert_to_tst_training_format, convert_to_test_instances


class LLMEvaluator(ModelEvaluator):
    """Evaluator for LLM-based bias detection models"""

    def __init__(self, trainable_predictor: TrainableLLMPredictor):
        """
        Initialize LLM evaluator.

        Args:
            trainable_predictor: Composed trainable predictor for training and inference
        """
        self.trainable_predictor = trainable_predictor

    def train_and_evaluate_fold(
        self, model: BiasModel, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, fold_num: int, seed: int
    ) -> float:
        """
        Evaluate LLM on a single fold with LoRA fine-tuning.

        Args:
            model: BiasModel (not actually used for LLM, just for interface compatibility)
            train_df: Training data for this fold
            test_df: Test data for this fold
            target_col: Name of the target column
            fold_num: Fold number for tracking
            seed: Random seed for reproducibility

        Returns:
            Accuracy score for this fold
        """
        try:
            # Convert training data to TsT format
            train_data = convert_to_tst_training_format(df=train_df, target_col=target_col, format_type=model.format)

            # Convert test data to test instances
            test_instances = convert_to_test_instances(df=test_df, target_col=target_col)

            # Train on this fold
            with tempfile.TemporaryDirectory() as temp_dir:
                adapter_dir = Path(temp_dir) / f"fold_{fold_num}"

                # Train adapter
                adapter_info = self.trainable_predictor.train(train_data, adapter_dir)

                # Load adapter for inference
                self.trainable_predictor.load_adapter(adapter_info.adapter_path)

                # Generate predictions
                predictions = self.trainable_predictor.predict(test_instances)

                # Calculate accuracy
                accuracy = self._calculate_accuracy(predictions, test_instances, model.format)

                return accuracy

        except Exception as e:
            # Ensure cleanup on error
            self.trainable_predictor.reset()
            raise RuntimeError(f"LLM evaluation failed on fold {fold_num}: {e}")

    def _calculate_accuracy(self, predictions: list, test_instances: list, format_type: Literal["mc", "num"]) -> float:
        """
        Calculate accuracy based on predictions and ground truth.

        Args:
            predictions: List of LLMPredictionResult objects
            test_instances: List of TestInstance objects
            format_type: "mc" for multiple choice, "num" for numerical

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        if not predictions or not test_instances:
            return 0.0

        # Create mapping from instance ID to ground truth
        ground_truth_map = {inst.instance_id: inst.ground_truth for inst in test_instances}

        correct = 0
        total = 0

        for pred in predictions:
            if pred.instance_id in ground_truth_map:
                gt = ground_truth_map[pred.instance_id]

                if format_type == "mc":
                    # Exact match for multiple choice
                    if pred.prediction.upper() == gt.upper():
                        correct += 1
                else:  # numerical
                    # For numerical, try to compare as floats with tolerance
                    try:
                        pred_val = float(pred.prediction)
                        gt_val = float(gt)
                        if abs(pred_val - gt_val) < 1e-6:
                            correct += 1
                    except (ValueError, TypeError):
                        # If conversion fails, fall back to string comparison
                        if pred.prediction.strip() == gt.strip():
                            correct += 1

                total += 1

        return correct / total if total > 0 else 0.0
