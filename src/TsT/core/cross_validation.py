"""
Unified cross-validation logic for all model types.

This module provides the common cross-validation framework that works with
any model type (RF, LLM, etc.) through the BiasModel protocol and ModelEvaluator interface.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from dataclasses import dataclass
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from ezcolorlog import root_logger as logger

from .protocols import BiasModel, ModelEvaluator
from .results import EvaluationResult, RepeatResult
from ..evaluators import RandomForestEvaluator, LLMEvaluator
from ..evaluators.llm.config import LLMRunConfig


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation"""

    n_folds: int = 5
    random_state: int = 42
    repeats: int = 1
    verbose: bool = True
    show_progress: bool = True


class UnifiedCrossValidator:
    """Unified cross-validation engine for all model types"""

    def __init__(self, config: Optional[CrossValidationConfig] = None):
        self.config = config or CrossValidationConfig()

    def __str__(self):
        return f"UnifiedCrossValidator(config={self.config})"

    def cross_validate(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str = "ground_truth",
        mode: Literal["rf", "llm"] = "rf",
        llm_config: Optional[LLMRunConfig] = None,
    ) -> EvaluationResult:
        resolved_target_col = self.resolve_target_column(model, target_col)
        match mode:
            case "rf":
                if llm_config is not None:
                    raise ValueError(f"RF mode does not require llm_config, got: {llm_config}")
                evaluator = RandomForestEvaluator()
            case "llm":
                evaluator = LLMEvaluator(model, df, resolved_target_col, llm_config)
            case _:
                raise ValueError(f"Unknown mode: {mode}")
        return self.run_cross_validation_repeats(model, evaluator, df, resolved_target_col)

    def run_cross_validation_repeats(
        self,
        model: BiasModel,
        evaluator: ModelEvaluator,
        df: pd.DataFrame,
        target_col: str = "ground_truth",
    ) -> EvaluationResult:
        """
        Run complete cross-validation evaluation for a model.

        Args:
            model: The bias model to evaluate
            evaluator: Fold evaluator for this model type
            df: Full dataset
            target_col: Target column name

        Returns:
            Complete evaluation result
        """
        # Select and prepare data
        qdf = model.select_rows(df)

        # Run repeated cross-validation
        repeat_results = []
        repeat_pbar = tqdm(
            range(self.config.repeats),
            desc=f"[{model.name.upper()}] Repeats",
            disable=self.config.repeats == 1 or not self.config.show_progress,
        )

        for repeat_id in repeat_pbar:
            repeat_result = self.run_cross_validation(model, evaluator, qdf, target_col, repeat_id)
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
        evaluation_result = evaluator.process_results(model, qdf, target_col, evaluation_result)

        # Log results
        if self.config.verbose:
            self.log_results(evaluation_result, self.config.n_folds, self.config.repeats)

        return evaluation_result

    def run_cross_validation(
        self,
        model: BiasModel,
        evaluator: ModelEvaluator,
        qdf: pd.DataFrame,
        target_col: str,
        repeat_id: int,
    ) -> RepeatResult:
        """Evaluate a single repeat (set of folds)"""
        seed = self.config.random_state + repeat_id

        # Create appropriate splitter
        if model.task == "reg":
            splitter = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=seed)
            split_args = (qdf,)
        else:
            splitter = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=seed)
            split_args = (qdf, qdf[target_col])

        # Evaluate folds
        fold_results = []
        fold_pbar = tqdm(
            enumerate(splitter.split(*split_args), 1),
            desc=f"[{model.name.upper()}] Folds",
            total=self.config.n_folds,
            disable=not self.config.show_progress or self.config.repeats > 1,
        )

        for fold_id, (train_idx, test_idx) in fold_pbar:
            train_df = qdf.iloc[train_idx].copy()
            test_df = qdf.iloc[test_idx].copy()

            # Evaluate fold
            fold_result = evaluator.train_and_evaluate_fold(model, train_df, test_df, target_col, fold_id, seed)
            fold_results.append(fold_result)

            # Update progress
            current_mean = np.mean([f.score for f in fold_results])
            fold_pbar.set_postfix({f"fold_{model.metric}": f"{current_mean:.2%}"})

        # Create repeat result
        return RepeatResult.from_fold_results(repeat_id, fold_results)

    @staticmethod
    def resolve_target_column(model: BiasModel, target_col: str) -> str:
        """Resolve target column with model-specific overrides"""
        if model.target_col_override is not None and model.target_col_override != target_col:
            logger.warning(
                f"[WARNING] {model.name} has an override target column '{model.target_col_override}'. Replacing '{target_col}'."
            )
            return model.target_col_override

        if model.task == "reg" and target_col == "gt_idx":
            logger.warning(
                f"[WARNING] {model.name} is numerical, with no gt_idx column. Overriding target column to 'ground_truth'"
            )
            return "ground_truth"

        return target_col

    @staticmethod
    def log_results(result: EvaluationResult, n_folds: int, repeats: int):
        """Log evaluation results"""
        logger.info(
            f"[{result.model_name.upper()}] "
            f"Overall {result.metric_name.upper()}: "
            f"{result.overall_mean:.2%} ± {result.overall_std:.2%} "
            f"(n_folds={n_folds}, repeats={repeats})"
        )

        if repeats == 1:
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

        # print the feature importances
        if result.feature_importances is not None:
            logger.info(f"[{result.model_name.upper()}] Feature importances:\n{result.feature_importances}")

        # print the model metadata
        if result.model_metadata is not None:
            logger.info(f"[{result.model_name.upper()}] Model metadata:\n{result.model_metadata}")
