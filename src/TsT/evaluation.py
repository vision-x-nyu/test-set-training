from typing import List, Union, Optional, Literal

import pandas as pd
from ezcolorlog import root_logger as logger

from .core.protocols import BiasModel
from .core.cross_validation import UnifiedCrossValidator, CrossValidationConfig
from .core.results import EvaluationResult
from .evaluators.llm.config import LLMRunConfig
from .utils import weighted_mean_std


# =============================================================================
# UNIFIED EVALUATION -----------------------------------------------------------
# =============================================================================


def run_evaluation(
    question_models: List[BiasModel],
    df_full: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False,
    repeats: int = 1,
    question_types: Union[List[str], None] = None,
    target_col: str = "ground_truth",
    mode: Literal["rf", "llm"] = "rf",
    llm_config: Optional[LLMRunConfig] = None,
) -> pd.DataFrame:
    """
    Run evaluation for all models and return a summary table of results.

    Supports both Random Forest and LLM-based bias detection with consistent
    cross-validation and detailed result reporting.

    Args:
        question_models: List of BiasModel models to evaluate (RF or LLM).
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
        question_models = [m for m in question_models if m.name in question_types]
        if not question_models:
            raise ValueError(f"Unknown question types: {question_types}")

    # Create cross-validator
    cv_config = CrossValidationConfig(
        n_folds=n_splits,
        random_state=random_state,
        repeats=repeats,
        verbose=verbose,
    )
    cross_validator = UnifiedCrossValidator(cv_config)

    # Validate evaluation mode
    if not question_models:
        raise ValueError("No models provided for evaluation")
    match mode:
        case "rf":
            if llm_config is not None:
                raise ValueError(f"RF mode does not require llm_config, got: {llm_config}")
        case "llm":
            if len(question_models) > 1:
                # TODO: make a single LLM model type
                raise ValueError("LLM evaluation only supports a single model")
            if llm_config is None:
                raise ValueError("llm_config is required for LLM evaluation")
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    # Evaluate all models
    results: List[EvaluationResult] = []
    for model in question_models:
        logger.error(f"\n================  {model.name.upper()}  ================\n\n")
        try:
            result = cross_validator.cross_validate(
                model=model,
                df=df_full,
                target_col=target_col,
                mode=mode,
                llm_config=llm_config,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Evaluation failed for {model.name}: {e}", exc_info=True)
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


# =============================================================================
# EVALUATION HELPER FUNCTIONS ------------------------------------------------
# =============================================================================


def _create_error_result(model: BiasModel, error_msg: str) -> EvaluationResult:
    """Create error result for failed evaluations"""
    from .core.results import EvaluationResult, RepeatResult, FoldResult

    # Create dummy fold and repeat results
    error_fold = FoldResult(fold_id=1, score=0.0, train_size=0, test_size=0, metadata={"error": error_msg})
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
