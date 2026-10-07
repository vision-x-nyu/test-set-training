"""
Unified k-fold cross-validation for all model types.

The cross-validator partitions a question type's rows into k disjoint folds,
trains on k-1 folds and scores the held-out fold, so every question is scored
exactly once per repeat by a model that never saw it. Model-specific training
and scoring live in a ModelEvaluator (Random Forest by default).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from tqdm.auto import tqdm

from .protocols import BiasModel, ModelEvaluator
from .protocols import EvaluationResult, RepeatResult

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation"""

    n_folds: int = 5
    random_state: int = 42
    repeats: int = 1
    verbose: bool = True
    show_progress: bool = True

    # Grouped folds: if set, rows sharing a value in this column (e.g. a scene,
    # video, or image id) are kept in the same fold. None = question-level folds.
    group_col: Optional[str] = None

    # Fold-assignment seed. None = random_state. When set, only the train/test
    # partition uses fold_seed + repeat index; the forest still uses random_state.
    fold_seed: Optional[int] = None


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
        target_col: Optional[str] = None,
        evaluator: Optional[ModelEvaluator] = None,
    ) -> EvaluationResult:
        """Run repeated k-fold cross-validation for one model.

        Args:
            model: The bias model to evaluate
            df: Full dataset (the model selects its own rows)
            target_col: Target column; None uses the model's default (see resolve_target_column)
            evaluator: Fold evaluator; defaults to the Random Forest evaluator (TsT-RF)
        """
        resolved_target_col = self.resolve_target_column(model, target_col)
        if evaluator is None:
            from ..evaluators.rf import RandomForestEvaluator

            evaluator = RandomForestEvaluator()
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

        # Evaluate folds
        fold_results = []
        fold_pbar = tqdm(
            enumerate(self.make_splits(model, qdf, target_col, self._split_seed(seed, repeat_id)), 1),
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

    def _split_seed(self, seed: int, repeat_id: int) -> int:
        """Seed for the fold partition: fold_seed + repeat_id if set, else the repeat seed."""
        return seed if self.config.fold_seed is None else self.config.fold_seed + repeat_id

    def make_splits(self, model: BiasModel, qdf: pd.DataFrame, target_col: str, seed: int):
        """Return the (train_idx, test_idx) positional splits for one repeat.

        Question-level folds by default (StratifiedKFold for classification, KFold for
        regression). With ``config.group_col`` set, rows sharing a group value stay in
        one fold (StratifiedGroupKFold / GroupKFold), and a split that puts a group in
        both train and test raises.
        """
        n_folds = self.config.n_folds
        group_col = self.config.group_col
        if group_col is None:
            if model.task == "reg":
                return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(qdf))
            return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(qdf, qdf[target_col]))

        if group_col not in qdf.columns:
            raise ValueError(f"group_col '{group_col}' not found in data for {model.name}")
        groups = qdf[group_col].astype(str).values
        if model.task == "reg":
            splits = list(GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(qdf, None, groups))
        else:
            splits = list(
                StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(
                    qdf, qdf[target_col], groups
                )
            )
        for fold_id, (train_idx, test_idx) in enumerate(splits, 1):
            overlap = set(groups[train_idx]) & set(groups[test_idx])
            logger.info(
                f"[{model.name.upper()}] grouped fold {fold_id}/{n_folds} by '{group_col}': "
                f"train={len(train_idx)} rows/{len(set(groups[train_idx]))} groups, "
                f"test={len(test_idx)} rows/{len(set(groups[test_idx]))} groups, "
                f"groups in both={len(overlap)}"
            )
            if overlap:
                raise RuntimeError(
                    f"Grouped fold {fold_id} leaks {len(overlap)} '{group_col}' groups across train/test"
                )
        return splits

    @staticmethod
    def resolve_target_column(model: BiasModel, target_col: Optional[str] = None) -> str:
        """Resolve the target column for one model.

        Precedence: the model's ``target_col_override`` (it always predicts that
        column), then an explicit ``target_col``, then the model's optional
        ``default_target_col`` attribute, then ``"ground_truth"``. Numerical
        models never use ``gt_idx``.
        """
        override = getattr(model, "target_col_override", None)
        if override is not None:
            if target_col is not None and target_col != override:
                logger.warning(f"{model.name} always predicts '{override}'; ignoring target column '{target_col}'.")
            return override

        if target_col is None:
            target_col = getattr(model, "default_target_col", None) or "ground_truth"

        if model.task == "reg" and target_col == "gt_idx":
            logger.warning(f"{model.name} is numerical and has no gt_idx; using 'ground_truth' as the target.")
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

        if result.feature_importances is not None:
            logger.info(f"[{result.model_name.upper()}] Feature importances:\n{result.feature_importances}")

        if result.model_metadata:
            logger.info(f"[{result.model_name.upper()}] Model metadata:\n{result.model_metadata}")
