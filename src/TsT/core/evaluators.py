"""
Model evaluators for different bias detection approaches.

This module contains evaluator classes that implement the actual evaluation
logic for different model types while working with the unified cross-validation framework.
"""

from typing import Dict, Any, Optional, List
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from ezcolorlog import root_logger as logger


from .protocols import ModelEvaluator, FeatureBasedBiasModel, BiasModel
from .results import FoldResult, EvaluationResult
from ..llm.data.models import TestInstance
from ..llm.predictors.vllm import VLLMPredictor, VLLMPredictorConfig
from ..llm.utils.llamafactory import (
    format_records_for_llama_factory_sft,
    generate_llama_factory_config,
    run_llama_factory_training,
)
from ..utils import mean_relative_accuracy


###############################################################################
# Random Forest Evaluator -----------------------------------------------------
###############################################################################


def make_rf_estimator(task, seed):
    if task == "clf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    else:
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """Label-encode *object* columns (fit on **train only** to avoid leak).
    Unseen categories in test are mapped to -1."""
    cat_cols = X_train.select_dtypes(include="object").columns
    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        enc = LabelEncoder().fit(X_train[col].astype(str))
        mapping = {cls: i for i, cls in enumerate(enc.classes_)}
        X_train[col] = X_train[col].astype(str).map(mapping).astype(int)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)
        encoders[col] = enc
    return encoders


def score_rf(est, X, y, metric="acc") -> float:
    if metric == "acc":
        return float(est.score(X, y))  # plain accuracy
    elif metric == "mra":
        y_pred = est.predict(X)
        return mean_relative_accuracy(y_pred, y.values.astype(float))
    else:
        raise ValueError(f"Unknown metric: {metric}")


class RandomForestEvaluator(ModelEvaluator):
    """RF model evaluator for unified evaluation framework"""

    def train_and_evaluate_fold(
        self,
        model: FeatureBasedBiasModel,  # Feature-based model
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate RF on a single fold and return rich result"""

        # Feature engineering
        model.fit_feature_maps(train_df)
        train_features = model.add_features(train_df)
        test_features = model.add_features(test_df)

        # Prepare data
        X_tr = train_features[model.feature_cols].copy()
        X_te = test_features[model.feature_cols].copy()
        encode_categoricals(X_tr, X_te)
        y_tr = train_features[target_col]
        y_te = test_features[target_col]

        # Train and evaluate
        estimator = make_rf_estimator(model.task, seed)
        estimator.fit(X_tr, y_tr)
        score = score_rf(estimator, X_te, y_te, model.metric)

        return FoldResult(
            fold_id=fold_id,
            score=score,
            train_size=len(train_df),
            test_size=len(test_df),
            metadata={
                "estimator_params": estimator.get_params(),
                "n_features": len(model.feature_cols),
            },
        )

    def process_results(
        self,
        model: FeatureBasedBiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Add feature importances to RF results"""

        # Train on full dataset for feature importances
        model.fit_feature_maps(df)
        full_df = model.add_features(df)
        X_full = full_df[model.feature_cols].copy()
        X_full_encoded = X_full.copy()
        encode_categoricals(X_full, X_full_encoded)
        y_full = full_df[target_col]

        # Use first repeat's first fold's seed for consistency
        seed = 42  # Default seed - could be made configurable
        if evaluation_result.repeat_results and evaluation_result.repeat_results[0].fold_results:
            # Try to extract seed from metadata if available
            pass

        rf_estimator = make_rf_estimator(model.task, seed)
        rf_estimator.fit(X_full_encoded, y_full)

        feature_importances = (
            pd.DataFrame({"feature": model.feature_cols, "importance": rf_estimator.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Update result
        evaluation_result.feature_importances = feature_importances
        evaluation_result.model_metadata.update(
            {
                "n_features": len(model.feature_cols),
                "feature_cols": model.feature_cols,
                "total_samples": len(df),
            }
        )

        return evaluation_result


###############################################################################
# LLM Evaluator ---------------------------------------------------------------
###############################################################################


def score_llm():
    raise NotImplementedError("LLM scoring not implemented")


def convert_to_blind_qa_format(df: pd.DataFrame, target_col: str, format_type: str) -> List[Dict[str, str]]:
    """
    Convert dataframe to blind QA format for LLM training.
    Removes all visual information, keeping only text-based questions and answers.
    """
    training_data = []

    for _, row in df.iterrows():
        # TODO: make this configurable
        # Extract question text - this varies by benchmark
        if "question" in row:
            question = row["question"]
        elif "instruction" in row:
            question = row["instruction"]
        elif "prompt" in row:
            question = row["prompt"]
        else:
            # Fallback: look for any text column that might contain the question
            text_cols = [col for col in row.index if isinstance(row[col], str) and len(str(row[col])) > 10]
            question = row[text_cols[0]] if text_cols else str(row.iloc[0])

        post_prompt = "Answer with the option's letter from the given choices directly."

        # Format based on question type
        if format_type == "mc":  # Multiple choice
            # Include answer choices in the question
            # TODO: make this configurable
            if "choices" in row:
                choices_text = " ".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(row["choices"])])
                instruction = f"{question} Choices: {choices_text}"
            elif "options" in row:
                choices_text = "\n".join(row["options"])
                instruction = f"{question} Options:\n{choices_text}"
            else:
                instruction = question

            instruction = f"{instruction}\n{post_prompt}"

            # Get the correct answer
            if target_col == "gt_idx":
                # Convert index to letter
                answer = chr(65 + int(row[target_col]))
            else:
                answer = str(row[target_col])

        else:  # Numerical or open-ended
            instruction = question
            answer = str(row[target_col])

        training_data.append(
            {
                "instruction": instruction,
                "response": answer,
            }
        )

    return training_data


def train_llm(train_data: List[Dict[str, str]], temp_path: Path, llm_config: Dict, fold: int, seed: int) -> str:
    """
    Train LLM on a single fold using LLaMA-Factory.
    Returns path to the trained adapter.
    """
    dataset_dir = temp_path / "dataset"
    output_dir = temp_path / "output" / f"fold_{fold}"

    # Format data for LLaMA-Factory
    sft_spec, dataset_path = format_records_for_llama_factory_sft(
        train_data,
        str(dataset_dir),
        instruction_key="instruction",
        response_key="response",
        overwrite=True,
    )

    # TODO: make this configurable
    # Add template to LLM config
    template = "gemma"
    assert llm_config["model_name"] == "google/gemma-2-2b-it", "Only Gemma is supported for now"

    # Generate training config
    config_path = generate_llama_factory_config(
        dataset_dir=str(dataset_dir),
        dataset_name=sft_spec.dataset_name,
        output_dir=str(output_dir),
        model_name=llm_config["model_name"],
        learning_rate=llm_config["learning_rate"],
        num_epochs=llm_config["num_epochs"],
        batch_size=llm_config["batch_size"],
        lora_rank=llm_config["lora_rank"],
        lora_alpha=llm_config["lora_alpha"],
        max_seq_length=llm_config["max_seq_length"],
        seed=seed,
        template=template,
    )

    # Run training
    run_llama_factory_training(config_path)

    return str(output_dir)


def evaluate_llm_zero_shot(predictor: VLLMPredictor, df: pd.DataFrame, target_col: str, format_type: str) -> float:
    """
    Evaluate zero-shot baseline performance.
    """
    logger.info(f"Evaluating zero-shot baseline for {format_type} format")
    test_data = convert_to_blind_qa_format(df, target_col, format_type)
    return evaluate_llm(predictor, test_data, format_type)


def evaluate_llm(predictor: VLLMPredictor, test_data: List[Dict[str, str]], format_type: str) -> float:
    """
    Evaluate LLM on test data and return accuracy.
    """
    instructions = [item["instruction"] for item in test_data]
    ground_truth = [item["response"] for item in test_data]

    # Convert to TestInstance format for the new API
    test_instances = [
        TestInstance(
            instruction=instruction,
            instance_id=f"eval_{i}",
            ground_truth=gt,
        )
        for i, (instruction, gt) in enumerate(zip(instructions, ground_truth))
    ]

    # TODO: make an evaluation function that takes LLMPredictionResult objects and returns a score
    # HACK [temporary]: manually score here
    # Generate predictions using new interface
    prediction_results = predictor.predict(test_instances)
    predictions = [result.prediction for result in prediction_results]

    # print the first prediction and ground truth
    if predictions and ground_truth:
        logger.info(f"First prediction: {predictions[0]}")
        logger.info(f"First ground truth: {ground_truth[0]}")

    # Calculate accuracy
    if format_type == "mc":
        # For multiple choice, exact match
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip().upper() == gt.strip().upper())
    else:
        # For numerical, use relative accuracy or exact match
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip() == gt.strip())

    return correct / len(test_data) if test_data else 0.0


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

        # TODO: evaluate baseline performance here?
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

        # Create training dataset in blind QA format
        train_data = convert_to_blind_qa_format(train_df, target_col, self.model.format)
        test_data = convert_to_blind_qa_format(test_df, target_col, self.model.format)

        # Fine-tune LLM on training fold
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # FT model on train fold
            adapter_path = train_llm(train_data, temp_path, self.llm_config, fold_id, seed)

            # eval on test fold
            self.predictor.load_adapter(adapter_path)  # load FTed adapter
            fold_score = evaluate_llm(self.predictor, test_data, self.model.format)

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
