"""
Unified cross-validation logic for all model types.

This module provides the common cross-validation framework that works with
any model type (RF, LLM, etc.) through the BiasModel protocol and ModelEvaluator interface.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from ezcolorlog import root_logger as logger

from .protocols import BiasModel, ModelEvaluator


def run_cross_validation(
    model: BiasModel,
    evaluator: ModelEvaluator,
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    repeats: int = 1,
    target_col: str = "ground_truth",
) -> Tuple[float, float, int]:
    """
    Common cross-validation logic for any model type.

    Args:
        model: Bias detection model implementing BiasModel protocol
        evaluator: Evaluator implementing ModelEvaluator interface
        df: Full dataset
        n_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        repeats: Number of times to repeat evaluation with different seeds
        target_col: Target column name

    Returns:
        Tuple of (mean_score, std_score, count)
    """
    # Select relevant rows for this model
    qdf = model.select_rows(df)
    all_scores = []

    # Handle target column override logic
    if model.target_col_override is not None:
        target_col = model.target_col_override
        if verbose:
            logger.info(f"Using target column override: {target_col}")

    if model.task == "reg" and target_col == "gt_idx":
        target_col = "ground_truth"
        if verbose:
            logger.warning(f"Model {model.name} is regression but target_col is 'gt_idx'. Using 'ground_truth'")

    # Progress tracking for repeats
    repeat_pbar = tqdm(range(repeats), desc=f"[{model.name.upper()}] Repeats", disable=repeats == 1)

    for repeat in repeat_pbar:
        current_seed = random_state + repeat

        # Create appropriate splitter based on task type
        if model.task == "reg":
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf,)
        else:  # classification
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf, qdf[target_col])

        scores = []
        fold_pbar = tqdm(
            enumerate(splitter.split(*split_args), 1),
            desc=f"[{model.name.upper()}] Folds",
            total=n_splits,
            disable=repeats > 1,
        )

        for fold, (tr_idx, te_idx) in fold_pbar:
            tr_df = qdf.iloc[tr_idx].copy()
            te_df = qdf.iloc[te_idx].copy()

            # Delegate fold evaluation to the evaluator
            fold_score = evaluator.evaluate_fold(model, tr_df, te_df, target_col, fold, current_seed)
            scores.append(fold_score)

            # Update progress bar
            current_mean = np.mean(scores)
            fold_pbar.set_postfix({f"fold_{model.metric}": f"{current_mean:.2%}"})

        all_scores.append(scores)
        if repeats > 1:
            current_avg = np.mean(scores)
            repeat_pbar.set_postfix({f"avg_{model.metric}": f"{current_avg:.2%}"})

    # Calculate statistics across all repeats
    mean_scores = [np.mean(scores) for scores in all_scores]
    mean_acc = float(np.mean(mean_scores))
    std_acc = float(np.std(mean_scores))
    count = len(qdf)

    # Log results if verbose
    if verbose:
        logger.info(
            f"[{model.name.upper()}] Overall {model.metric.upper()}: "
            f"{mean_acc:.2%} ± {std_acc:.2%} "
            f"(n_splits={n_splits}, repeats={repeats})"
        )
        if repeats == 1:
            fold_scores = [f"{s:.2%}" for s in all_scores[0]]
            logger.info(f"[{model.name.upper()}] Fold scores: {fold_scores}")
        else:
            repeat_scores = [f"{s:.2%}" for s in mean_scores]
            logger.info(f"[{model.name.upper()}] Repeat scores: {repeat_scores}")

    return mean_acc, std_acc, count
