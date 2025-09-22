from typing import List, Union, Optional, Literal, Dict
import dataclasses

import numpy as np
import pandas as pd
from ezcolorlog import root_logger as logger
from scipy.stats import t

from .core.protocols import BiasModel
from .core.cross_validation import UnifiedCrossValidator, CrossValidationConfig
from .core.protocols import EvaluationResult
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

    # Calculate and log overall statistics
    log_overall_statistics(results)

    return results


# =============================================================================
# EVALUATION HELPER FUNCTIONS ------------------------------------------------
# =============================================================================


def _create_error_result(model: BiasModel, error_msg: str) -> EvaluationResult:
    """Create error result for failed evaluations"""
    from .core.protocols import EvaluationResult, RepeatResult, FoldResult

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


def get_overall_eval_stats(results: List[EvaluationResult]) -> Dict[str, float]:
    """Get overall statistics across all evaluation results.

    Computes both macro and micro averages:
    - 'macro': Averaging the metric independently for each model and then taking the
      arithmetic mean, giving equal weight to each model.
    - 'micro': Aggregating the contributions of all models to compute the metric globally,
      treating every instance equally—this is effectively a global weighted mean over
      the entire dataset.

    Uncertainty Measures:
    - 'se': Standard Error of the mean
    - 't_95ci': 95% t-confidence interval
    """
    if not results or len(results) == 0:
        logger.warning("No evaluation results to summarize")
        return {}

    # Convert percentage strings back to float for calculations

    # assert that all evaluation results have the same number of repeats
    num_repeats = results[0].repeats
    assert all(r.repeats == results[0].repeats for r in results), (
        f"All evaluation results must have the same number of repeats, got {dataclasses.asdict(results)}"
    )

    repeat_scores = np.array([r.repeat_scores for r in results])  # (models, repeats)
    repeat_scores_macro_avg = repeat_scores.mean(axis=0)  # (repeats,)
    print(f"repeat_scores_macro_avg: {repeat_scores_macro_avg}")

    # calculate macro stats over repeats
    score_macro_avg = repeat_scores_macro_avg.mean()
    if num_repeats < 2:
        logger.warning(f"got {num_repeats} repeats. std, se, & t-CI are ill-defined for <2 repeats")
        score_macro_std = np.nan
        score_macro_se = np.nan
        score_macro_t_95ci = (np.nan, np.nan)
    else:
        score_macro_std = repeat_scores_macro_avg.std(ddof=1)  # sample SD (unbiased)
        score_macro_se = score_macro_std / np.sqrt(repeat_scores_macro_avg.shape[0])  # sample Std Error of the mean
        score_macro_t_95ci = t.interval(
            0.95, repeat_scores_macro_avg.shape[0] - 1, loc=score_macro_avg, scale=score_macro_se
        )

    # NOTE: zs baseline is computed once per model. is not repeated. cannot run t-CI on it.
    zs_baselines = np.array([r.zero_shot_baseline for r in results])
    zs_macro_avg = zs_baselines.mean()

    # todo: do "micro" average properly
    scores = np.array([r.overall_mean for r in results])
    counts = np.array([r.count for r in results])
    total_count = counts.sum()
    weighted_avg, weighted_std = weighted_mean_std(scores, counts)
    zs_weighted_avg, zs_weighted_std = weighted_mean_std(zs_baselines, counts)

    return dict(
        total_count=int(total_count),
        score_macro=dict(
            n=repeat_scores_macro_avg.shape[0],
            avg=float(score_macro_avg),
            std=float(score_macro_std),
            se=float(score_macro_se),
            t_95ci=(float(score_macro_t_95ci[0]), float(score_macro_t_95ci[1])),
        ),
        zs_macro_avg=float(zs_macro_avg),
        # TODO: micro
        weighted_avg=float(weighted_avg),
        weighted_std=float(weighted_std),
        zs_weighted_avg=float(zs_weighted_avg),
        zs_weighted_std=float(zs_weighted_std),
    )


def log_overall_statistics(results: List[EvaluationResult]):
    """Log overall evaluation statistics"""

    overall_stats = get_overall_eval_stats(results)
    total_count = overall_stats["total_count"]
    score_macro = overall_stats["score_macro"]
    zs_macro_avg = overall_stats["zs_macro_avg"]
    weighted_avg = overall_stats["weighted_avg"]
    weighted_std = overall_stats["weighted_std"]
    zs_weighted_avg = overall_stats["zs_weighted_avg"]
    zs_weighted_std = overall_stats["zs_weighted_std"]

    summary_df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    summary_df = summary_df.sort_values("overall_mean", ascending=False)
    summary_df = summary_df.rename(
        columns={
            "model_name": "Model",
            "model_format": "Format",
            "metric_name": "Metric",
            "overall_mean": "Score",
            "overall_std": "Std Dev",
            "count": "Count",
            "repeats": "Repeats",
            "zero_shot_baseline": "ZS Baseline",
        }
    )

    # Print pretty table
    table_summary = "\n" * 3 + "=" * 80 + "\n"
    table_summary += "UNIFIED EVALUATION SUMMARY\n"
    table_summary += "=" * 80 + "\n"
    table_summary += (
        summary_df[["Model", "Format", "Metric", "Score", "Std Dev", "Count", "Repeats", "ZS Baseline"]].to_string(
            index=False
        )
        + "\n"
    )
    table_summary += "=" * 80 + "\n"
    table_summary += f"""SCORE MACRO AVERAGE: {score_macro["avg"]:.1%}
    n: {score_macro["n"]}
    std: {score_macro["std"]:.1%}
    se: {score_macro["se"]:.1%}
    95% t-CI: ({score_macro["t_95ci"][0]:.1%}, {score_macro["t_95ci"][1]:.1%})
"""
    table_summary += f"ZERO-SHOT BASELINE MACRO AVERAGE: {zs_macro_avg:.1%}\n"
    table_summary += f"WEIGHTED MEAN SCORE: {weighted_avg:.1%} ± {weighted_std:.1%} (total examples: {total_count})\n"
    table_summary += f"WEIGHTED ZERO-SHOT BASELINE: {zs_weighted_avg:.1%} ± {zs_weighted_std:.1%}\n"
    table_summary += "=" * 80 + "\n"
    logger.info(table_summary)
