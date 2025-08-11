import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Literal
import tempfile
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from ezcolorlog import root_logger as logger

from .core.protocols import BiasModel, FeatureBasedBiasModel
from .core.cross_validation import UnifiedCrossValidator, CrossValidationConfig
from .core.evaluators import RandomForestFoldEvaluator, RandomForestPostProcessor, LLMFoldEvaluator, LLMPostProcessor
from .core.results import EvaluationResult
from .llm.utils.llamafactory import (
    format_records_for_llama_factory_sft,
    generate_llama_factory_config,
    run_llama_factory_training,
)
from .llm.predictors.vllm import VLLMPredictor, VLLMPredictorConfig
from .llm.data.models import TestInstance


# =============================================================================
# HELPERS ----------------------------------------------------------------------
# =============================================================================


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """Label‑encode *object* columns (fit on **train only** to avoid leak).
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


def mean_relative_accuracy(pred, true, start=0.5, end=0.95, step=0.05):
    thresholds = np.linspace(start, end, int((end - start) / step) + 2)
    rel_err = np.abs(pred - true) / true
    return np.mean([(rel_err < 1 - t).mean() for t in thresholds])


def _make_estimator(task, seed):
    if task == "clf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    else:
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def _score(est, X, y, metric="acc"):
    if metric == "acc":
        return est.score(X, y)  # plain accuracy
    elif metric == "mra":
        y_pred = est.predict(X)
        return mean_relative_accuracy(y_pred, y.values.astype(float))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _generate_feature_importances(
    model: FeatureBasedBiasModel, df: pd.DataFrame, target_col: str, random_state: int
) -> pd.DataFrame:
    """Generate feature importances for Random Forest models"""
    # Train on full dataset for feature importances
    qdf = model.select_rows(df)
    model.fit_feature_maps(qdf)
    full_df = model.add_features(qdf)
    X_full = full_df[model.feature_cols].copy()
    encode_categoricals(X_full, X_full.copy())
    y_full = full_df[target_col]

    estimator = _make_estimator(model.task, random_state)
    estimator.fit(X_full, y_full)

    feature_importances = (
        pd.DataFrame({"feature": model.feature_cols, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return feature_importances


def _generate_llm_feature_importances(mean_score: float) -> pd.DataFrame:
    """Generate mock feature importances for LLM models for compatibility"""
    return pd.DataFrame({"feature": ["llm_finetuning", "improvement"], "importance": [mean_score, mean_score]})


def weighted_mean_std(scores: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """
    Weighted mean:
        weighted_avg = sum(score_i * count_i) / sum(count_i)
    Weighted variance:
        weighted_var = sum(count_i * (score_i - weighted_avg)**2) / sum(count_i)
    Weighted std:
        weighted_std = sqrt(weighted_var)
    Only models with count > 0 are included.
    """
    mask = counts > 0
    scores = scores[mask]
    counts = counts[mask]
    weighted_avg = (scores * counts).sum() / counts.sum() if counts.sum() > 0 else 0
    weighted_var = ((counts * (scores - weighted_avg) ** 2).sum() / counts.sum()) if counts.sum() > 0 else 0
    weighted_std = weighted_var**0.5
    return weighted_avg, weighted_std


# =============================================================================
# UNIFIED EVALUATION -----------------------------------------------------------
# =============================================================================


def run_evaluation(
    models: List[BiasModel],
    df_full: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False,
    repeats: int = 1,
    question_types: Union[List[str], None] = None,
    target_col: str = "ground_truth",
    mode: Literal["rf", "llm"] = "rf",
    llm_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Run evaluation for all models and return a summary table of results.

    Supports both Random Forest and LLM-based bias detection with consistent
    cross-validation and detailed result reporting.

    Args:
        models: List of BiasModel models to evaluate (RF or LLM).
        df_full: The full dataframe containing all data.
        n_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed output during evaluation
        repeats: Number of times to repeat evaluation with different random seeds
        question_types: Optional list of question types to evaluate. If None, evaluate all types.
        target_col: Column to use as target variable (default: "ground_truth")
        mode: Evaluation mode - "rf" for Random Forest, "llm" for LLM-based evaluation
        llm_config: Configuration dict for LLM mode (model_name, batch_size, etc.)

    Returns:
        DataFrame with model results including mean score and standard deviation
    """
    # Filter models if question_types is specified
    if question_types is not None:
        models = [m for m in models if m.name in question_types]
        if not models:
            raise ValueError(f"Unknown question types: {question_types}")

    # Create cross-validator
    cv_config = CrossValidationConfig(
        n_splits=n_splits,
        random_state=random_state,
        repeats=repeats,
        verbose=verbose,
    )
    cross_validator = UnifiedCrossValidator(cv_config)

    # Create evaluator and post-processor based on mode
    if mode == "rf":
        evaluator = RandomForestFoldEvaluator()
        post_processor = RandomForestPostProcessor()
    elif mode == "llm":
        # Get zero-shot baseline first
        if not models:
            raise ValueError("No models provided for LLM evaluation")
        zero_shot_baseline = _get_zero_shot_baseline(models[0], df_full, target_col, llm_config or {})
        evaluator = LLMFoldEvaluator(llm_config or {})
        post_processor = LLMPostProcessor(llm_config or {}, zero_shot_baseline)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Evaluate all models
    results: List[EvaluationResult] = []
    for model in models:
        logger.info(f"\n================  {model.name.upper()}  ================")
        try:
            result = cross_validator.evaluate_model(
                model=model,
                evaluator=evaluator,
                df=df_full,
                target_col=target_col,
                post_processor=post_processor,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Evaluation failed for {model.name}: {e}")
            # Create error result
            results.append(_create_error_result(model, str(e)))

    # Convert to summary DataFrame
    summary_data = [r.to_summary_dict() for r in results]
    summary = pd.DataFrame(summary_data)

    # Sort by score (need to convert percentage strings back to float for sorting)
    summary["_sort_score"] = summary["Score"].str.rstrip("%").astype(float) / 100
    summary = summary.sort_values("_sort_score", ascending=False).drop("_sort_score", axis=1)

    # Calculate and log overall statistics
    _log_overall_statistics(summary, results)

    return summary


def evaluate_bias_model_llm(
    model: FeatureBasedBiasModel,
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    repeats: int = 1,
    target_col: str = "ground_truth",
    llm_config: Optional[Dict] = None,
):
    """
    Evaluate bias model using LLM fine-tuning instead of Random Forest.

    This implements the TsT method with LLM fine-tuning as described in the task document.
    Uses LoRA fine-tuning on k-fold splits to learn non-visual shortcuts.
    """
    if llm_config is None:
        llm_config = {
            "model_name": "google/gemma-2-2b-it",
            "batch_size": 32,
            "learning_rate": 2e-4,
            "num_epochs": 5,
            "lora_rank": 8,
            "lora_alpha": 16,
            "max_seq_length": 512,
        }

    qdf = model.select_rows(df)
    all_scores = []
    all_zero_shot_scores = []

    # Check target column override
    if model.target_col_override is not None and model.target_col_override != target_col:
        logger.warning(
            f"[WARNING] {model.name} has an override target column '{model.target_col_override}'. Replacing '{target_col}'."
        )
        target_col = model.target_col_override
    if model.task == "reg" and target_col == "gt_idx":
        target_col = "ground_truth"
        logger.warning(
            f"[WARNING] {model.name} is numerical, with no gt_idx column. Overriding target column to 'ground_truth'"
        )

    # Initialize LLM predictor
    llm_config_obj = VLLMPredictorConfig(
        model_name=llm_config["model_name"],
        batch_size=llm_config["batch_size"],
        max_seq_length=llm_config["max_seq_length"],
        apply_chat_template=False,  # Disable for compatibility with Gemma and other models
    )
    llm_predictor = VLLMPredictor(llm_config_obj)

    # Get zero-shot baseline first
    zero_shot_acc = _evaluate_zero_shot_baseline(llm_predictor, qdf, target_col, model.format)
    logger.info(f"Zero-shot baseline accuracy: {zero_shot_acc:.2%}")
    llm_predictor.reset()  # cleanup the base model after getting the zero-shot baseline

    repeat_pbar = tqdm(range(repeats), desc=f"[{model.name.upper()}] LLM Repeats", disable=repeats == 1)
    for repeat in repeat_pbar:
        current_seed = random_state + repeat

        # Use appropriate splitter based on task type
        if model.task == "reg":
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf,)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf, qdf[target_col])

        scores = []
        zero_shot_scores = []

        fold_pbar = tqdm(
            enumerate(splitter.split(*split_args), 1),
            desc=f"[{model.name.upper()}] LLM Folds",
            total=n_splits,
            disable=repeats > 1,
        )

        for fold, (tr_idx, te_idx) in fold_pbar:
            tr, te = qdf.iloc[tr_idx].copy(), qdf.iloc[te_idx].copy()

            # Create training dataset in blind QA format
            train_data = _convert_to_blind_qa_format(tr, target_col, model.format)
            test_data = _convert_to_blind_qa_format(te, target_col, model.format)

            # Fine-tune LLM on training fold
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Fine-tune model
                adapter_path = _train_llm_fold(train_data, temp_path, llm_config, fold, current_seed)

                # Load fine-tuned adapter
                llm_predictor.load_adapter(adapter_path)

                # Evaluate on test fold
                fold_score = _evaluate_llm_fold(llm_predictor, test_data, model.format)
                scores.append(fold_score)

                # Also track zero-shot for comparison
                zero_shot_scores.append(zero_shot_acc)

                fold_pbar.set_postfix(
                    {"fold_acc": f"{np.mean(scores):.2%}", "vs_zero_shot": f"+{np.mean(scores) - zero_shot_acc:.2%}"}
                )

        all_scores.append(scores)
        all_zero_shot_scores.append(zero_shot_scores)

        if repeats > 1:
            current_avg = np.mean(scores)
            repeat_pbar.set_postfix({"avg_acc": f"{current_avg:.2%}"})

    # Calculate statistics
    mean_scores = [np.mean(scores) for scores in all_scores]
    mean_acc = float(np.mean(mean_scores))
    std_acc = float(np.std(mean_scores))
    count = len(qdf)

    # Calculate improvement over zero-shot
    improvement = mean_acc - zero_shot_acc

    if verbose:
        logger.info(f"\n[{model.name.upper()}] LLM TsT Results:")
        logger.info(f"Zero-shot baseline: {zero_shot_acc:.2%}")
        logger.info(f"TsT-LoRA accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
        logger.info(f"Improvement: +{improvement:.2%}")
        if repeats == 1:
            logger.info(f"[{model.name.upper()}] Fold accuracies: {[f'{s:.2%}' for s in all_scores[0]]}")
        else:
            logger.info(f"[{model.name.upper()}] Repeat accuracies: {[f'{s:.2%}' for s in mean_scores]}")

    # Create mock feature importances for compatibility
    fi = pd.DataFrame(
        {
            "feature": ["llm_finetuning", "zero_shot_baseline", "improvement"],
            "importance": [mean_acc, zero_shot_acc, improvement],
        }
    )

    return mean_acc, std_acc, fi, count


def _convert_to_blind_qa_format(df: pd.DataFrame, target_col: str, format_type: str) -> List[Dict[str, str]]:
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


def _train_llm_fold(train_data: List[Dict[str, str]], temp_path: Path, llm_config: Dict, fold: int, seed: int) -> str:
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


def _evaluate_zero_shot_baseline(predictor, df: pd.DataFrame, target_col: str, format_type: str) -> float:
    """
    Evaluate zero-shot baseline performance.
    """
    logger.info(f"Evaluating zero-shot baseline for {format_type} format")
    test_data = _convert_to_blind_qa_format(df, target_col, format_type)
    return _evaluate_llm_fold(predictor, test_data, format_type)


def _evaluate_llm_fold(predictor: VLLMPredictor, test_data: List[Dict[str, str]], format_type: str) -> float:
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


# =============================================================================
# EVALUATION HELPER FUNCTIONS ------------------------------------------------
# =============================================================================


def _get_zero_shot_baseline(model: BiasModel, df: pd.DataFrame, target_col: str, llm_config: Dict) -> float:
    """Get zero-shot baseline for LLM evaluation"""
    # TODO: inference + evaluate the model on the full testset before any training
    raise NotImplementedError(
        "Zero-shot baseline evaluation not yet implemented. This requires implementing actual LLM zero-shot inference."
    )


def _create_error_result(model: BiasModel, error_msg: str) -> EvaluationResult:
    """Create error result for failed evaluations"""
    from .core.results import EvaluationResult, RepeatResult, FoldResult

    # Create dummy fold and repeat results
    error_fold = FoldResult(fold_id=1, score=0.0, fold_size=0, metadata={"error": error_msg})
    error_repeat = RepeatResult.from_fold_results(0, [error_fold])

    return EvaluationResult.from_repeat_results(
        model_name=model.name,
        model_format=getattr(model, "format", "unknown"),
        metric_name=getattr(model, "metric", "unknown"),
        repeat_results=[error_repeat],
        model_metadata={"error": error_msg},
    )


def _log_overall_statistics(summary: pd.DataFrame, results: List[EvaluationResult]):
    """Log overall evaluation statistics"""
    if summary.empty:
        logger.warning("No evaluation results to summarize")
        return

    # Convert percentage strings back to float for calculations
    scores = summary["Score"].str.rstrip("%").astype(float) / 100
    counts = summary["Count"]

    # Calculate overall statistics
    overall_avg = scores.mean()
    overall_std = scores.std()
    total_count = counts.sum()

    # Weighted mean and std calculation
    weighted_avg, weighted_std = weighted_mean_std(scores.values, counts.values)

    # Print pretty table
    table_summary = "\n" * 3 + "=" * 80 + "\n"
    table_summary += "UNIFIED EVALUATION SUMMARY\n"
    table_summary += "=" * 80 + "\n"
    table_summary += summary[["Model", "Format", "Metric", "Score", "± Std", "Count"]].to_string(index=False) + "\n"
    table_summary += "=" * 80 + "\n"
    table_summary += f"OVERALL AVERAGE SCORE: {overall_avg:.1%} ± {overall_std:.1%}\n"
    table_summary += (
        f"WEIGHTED AVERAGE SCORE: {weighted_avg:.1%} ± {weighted_std:.1%} (total examples: {total_count})\n"
    )
    table_summary += "=" * 80 + "\n"
    logger.info(table_summary)
