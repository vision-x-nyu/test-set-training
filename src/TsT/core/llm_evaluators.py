"""
LLM evaluators for the unified evaluation framework.

This module extends the Phase 1 evaluation system to support LLM-based bias detection,
integrating with the new LLM infrastructure while maintaining compatibility with
the existing evaluation pipeline.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Literal
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

    def evaluate_fold(
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


class TemporaryLLMEvaluator(ModelEvaluator):
    """
    Temporary LLM evaluator for backward compatibility.

    This bridges the old evaluate_bias_model_llm function until the full
    LLM infrastructure is ready. It maintains the existing LLM functionality
    during the Phase 1 -> Phase 2 transition.
    """

    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize temporary LLM evaluator.

        Args:
            llm_config: Configuration dictionary for LLM evaluation
        """
        self.llm_config = llm_config

    def evaluate_fold(
        self, model: BiasModel, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, fold_num: int, seed: int
    ) -> float:
        """
        Evaluate using the legacy LLM evaluation function.

        This is a temporary bridge that calls the existing evaluate_bias_model_llm
        logic until the full Phase 2 LLM system is integrated.
        """
        # Import here to avoid circular imports
        from ..evaluation import evaluate_bias_model_llm

        # Create a temporary combined DataFrame for the legacy function
        train_df_copy = train_df.copy()
        train_df_copy["split"] = "train"
        test_df_copy = test_df.copy()
        test_df_copy["split"] = "test"

        combined_df = pd.concat([train_df_copy, test_df_copy], ignore_index=True)

        # Call the legacy function with n_splits=1 to use our pre-split data
        mean_score, _, _ = evaluate_bias_model_llm(
            model=model,
            df=combined_df,
            n_splits=1,  # Use pre-split data
            random_state=seed,
            verbose=False,
            repeats=1,
            target_col=target_col,
            llm_model=self.llm_config.get("model_name", "google/gemma-2-2b-it"),
            llm_epochs=self.llm_config.get("epochs", 1),
            llm_batch_size=self.llm_config.get("batch_size", 4),
        )

        return mean_score


def create_llm_evaluator(
    trainable_predictor: Optional[TrainableLLMPredictor] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    use_legacy: bool = False,
) -> ModelEvaluator:
    """
    Factory function to create appropriate LLM evaluator.

    Args:
        trainable_predictor: Full trainable predictor (for Phase 2)
        llm_config: Configuration for legacy LLM evaluation
        use_legacy: Whether to use the legacy evaluator

    Returns:
        Appropriate LLM evaluator instance
    """
    if use_legacy or trainable_predictor is None:
        if llm_config is None:
            llm_config = {
                "model_name": "google/gemma-2-2b-it",
                "epochs": 1,
                "batch_size": 4,
            }
        return TemporaryLLMEvaluator(llm_config)
    else:
        return LLMEvaluator(trainable_predictor)
