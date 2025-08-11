"""
Unified cross-validation logic for all model types.

This module provides the common cross-validation framework that works with
any model type (RF, LLM, etc.) through the BiasModel protocol and ModelEvaluator interface.
"""

import numpy as np
import pandas as pd
from typing import Protocol, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from ezcolorlog import root_logger as logger

from .protocols import BiasModel
from .results import EvaluationResult, RepeatResult, FoldResult


class FoldEvaluator(Protocol):
    """Protocol for evaluating a single fold with rich result objects"""

    def evaluate_fold(
        self,
        model: BiasModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate a single fold and return detailed result"""
        ...


class PostProcessor(Protocol):
    """Protocol for post-processing evaluation results"""

    def process_results(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Post-process results (e.g., add feature importances)"""
        ...


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation"""

    n_splits: int = 5
    random_state: int = 42
    repeats: int = 1
    verbose: bool = True
    show_progress: bool = True


class UnifiedCrossValidator:
    """Unified cross-validation engine for all model types"""

    def __init__(self, config: Optional[CrossValidationConfig] = None):
        self.config = config or CrossValidationConfig()

    def evaluate_model(
        self,
        model: BiasModel,
        evaluator: FoldEvaluator,
        df: pd.DataFrame,
        target_col: str = "ground_truth",
        post_processor: Optional[PostProcessor] = None,
    ) -> EvaluationResult:
        """
        Run complete cross-validation evaluation for a model.

        Args:
            model: The bias model to evaluate
            evaluator: Fold evaluator for this model type
            df: Full dataset
            target_col: Target column name
            post_processor: Optional post-processing (e.g., feature importances)

        Returns:
            Complete evaluation result
        """
        # Select and prepare data
        qdf = model.select_rows(df)
        target_col = self._resolve_target_column(model, target_col)

        # Run repeated cross-validation
        repeat_results = []
        repeat_pbar = tqdm(
            range(self.config.repeats),
            desc=f"[{model.name.upper()}] Repeats",
            disable=self.config.repeats == 1 or not self.config.show_progress,
        )

        for repeat_id in repeat_pbar:
            repeat_result = self._evaluate_repeat(model, evaluator, qdf, target_col, repeat_id)
            repeat_results.append(repeat_result)

            if self.config.repeats > 1:
                repeat_pbar.set_postfix({f"avg_{model.metric}": f"{repeat_result.mean_score:.2%}"})

        # Create evaluation result
        evaluation_result = EvaluationResult.from_repeat_results(
            model_name=model.name,
            model_format=model.format,
            metric_name=model.metric,
            repeat_results=repeat_results,
        )

        # Post-process if needed (e.g., feature importances)
        if post_processor is not None:
            evaluation_result = post_processor.process_results(model, qdf, target_col, evaluation_result)

        # Log results
        if self.config.verbose:
            self._log_results(evaluation_result)

        return evaluation_result

    def _evaluate_repeat(
        self,
        model: BiasModel,
        evaluator: FoldEvaluator,
        qdf: pd.DataFrame,
        target_col: str,
        repeat_id: int,
    ) -> RepeatResult:
        """Evaluate a single repeat (set of folds)"""
        seed = self.config.random_state + repeat_id

        # Create appropriate splitter
        if model.task == "reg":
            splitter = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=seed)
            split_args = (qdf,)
        else:
            splitter = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=seed)
            split_args = (qdf, qdf[target_col])

        # Evaluate folds
        fold_results = []
        fold_pbar = tqdm(
            enumerate(splitter.split(*split_args), 1),
            desc=f"[{model.name.upper()}] Folds",
            total=self.config.n_splits,
            disable=self.config.repeats > 1 or not self.config.show_progress,
        )

        for fold_id, (tr_idx, te_idx) in fold_pbar:
            tr_df = qdf.iloc[tr_idx].copy()
            te_df = qdf.iloc[te_idx].copy()

            # Evaluate fold
            fold_result = evaluator.evaluate_fold(model, tr_df, te_df, target_col, fold_id, seed)
            fold_results.append(fold_result)

            # Update progress
            current_mean = np.mean([f.score for f in fold_results])
            fold_pbar.set_postfix({f"fold_{model.metric}": f"{current_mean:.2%}"})

        # Create repeat result
        return RepeatResult.from_fold_results(repeat_id, fold_results)

    def _resolve_target_column(self, model: BiasModel, target_col: str) -> str:
        """Resolve target column with model-specific overrides"""
        if model.target_col_override is not None:
            return model.target_col_override

        if model.task == "reg" and target_col == "gt_idx":
            return "ground_truth"

        return target_col

    def _log_results(self, result: EvaluationResult):
        """Log evaluation results"""
        logger.info(
            f"[{result.model_name.upper()}] "
            f"Overall {result.metric_name.upper()}: "
            f"{result.overall_mean:.2%} ± {result.overall_std:.2%} "
            f"(n_splits={self.config.n_splits}, repeats={self.config.repeats})"
        )

        if self.config.repeats == 1:
            fold_scores = [f.score for f in result.repeat_results[0].fold_results]
            logger.info(
                f"[{result.model_name.upper()}] Fold {result.metric_name.upper()}s: {[f'{s:.2%}' for s in fold_scores]}"
            )
        else:
            repeat_scores = [r.mean_score for r in result.repeat_results]
            logger.info(
                f"[{result.model_name.upper()}] "
                f"Repeat {result.metric_name.upper()}s: "
                f"{[f'{s:.2%}' for s in repeat_scores]}"
            )
