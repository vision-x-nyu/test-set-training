# Phase 3: Unified Evaluation Framework

## Background

After Phase 1 created base abstractions and Phase 2 implemented the LLM system, we now have two parallel evaluation pipelines (RF and LLM) that share common cross-validation logic but handle it differently. This creates maintenance overhead and potential inconsistencies.

Current issues:
1. **Duplicated CV Logic**: Similar cross-validation patterns in both RF and LLM evaluators
2. **Inconsistent Reporting**: Different progress tracking, logging, and result formatting
3. **Hard to Extend**: Adding new model types requires duplicating CV boilerplate
4. **Testing Complexity**: Need to test CV logic in multiple places

## Objectives

Create a unified evaluation framework that handles cross-validation, progress tracking, result aggregation, and reporting consistently across all model types (RF, LLM, and future extensions).

## Implementation Plan

### 1. Unified Evaluation Results

**File**: `src/TsT/core/results.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel

@dataclass
class FoldResult:
    """Result from a single CV fold"""
    fold_id: int
    score: float
    fold_size: int
    metadata: Dict[str, Any] = None

@dataclass  
class RepeatResult:
    """Result from a single repeat (collection of folds)"""
    repeat_id: int
    fold_results: List[FoldResult]
    mean_score: float
    std_score: float
    
    @property
    def total_instances(self) -> int:
        return sum(f.fold_size for f in self.fold_results)

@dataclass
class EvaluationResult:
    """Complete evaluation result for a model"""
    model_name: str
    model_format: str
    metric_name: str
    repeat_results: List[RepeatResult]
    
    # Aggregated statistics
    overall_mean: float
    overall_std: float
    total_count: int
    
    # Model-specific metadata
    feature_importances: Optional[pd.DataFrame] = None
    model_metadata: Dict[str, Any] = None
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for summary table"""
        return {
            "Model": self.model_name,
            "Format": self.model_format.upper(),
            "Metric": self.metric_name.upper(),
            "Score": f"{self.overall_mean:.1%}",
            "± Std": f"{self.overall_std:.1%}",
            "Count": self.total_count,
            "Feature Importances": self.feature_importances,
            "Metadata": self.model_metadata,
        }
```

### 2. Generic Cross-Validation Engine

**File**: `src/TsT/core/cross_validation.py`

```python
from abc import ABC, abstractmethod
from typing import List, Protocol, Dict, Any, Optional, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm.auto import tqdm

from .protocols import BiasModel
from .results import EvaluationResult, RepeatResult, FoldResult

class FoldEvaluator(Protocol):
    """Protocol for evaluating a single fold"""
    
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
    
    def __init__(self, config: CrossValidationConfig = None):
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
            repeat_result = self._evaluate_repeat(
                model, evaluator, qdf, target_col, repeat_id
            )
            repeat_results.append(repeat_result)
            
            if self.config.repeats > 1:
                repeat_pbar.set_postfix({
                    f"avg_{model.metric}": f"{repeat_result.mean_score:.2%}"
                })
        
        # Aggregate results
        overall_mean, overall_std, total_count = self._aggregate_results(repeat_results)
        
        evaluation_result = EvaluationResult(
            model_name=model.name,
            model_format=model.format,
            metric_name=model.metric,
            repeat_results=repeat_results,
            overall_mean=overall_mean,
            overall_std=overall_std,
            total_count=total_count,
        )
        
        # Post-process if needed (e.g., feature importances)
        if post_processor is not None:
            evaluation_result = post_processor.process_results(
                model, qdf, target_col, evaluation_result
            )
        
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
            fold_result = evaluator.evaluate_fold(
                model, tr_df, te_df, target_col, fold_id, seed
            )
            fold_results.append(fold_result)
            
            # Update progress
            current_mean = np.mean([f.score for f in fold_results])
            fold_pbar.set_postfix({f"fold_{model.metric}": f"{current_mean:.2%}"})
        
        # Calculate repeat statistics
        scores = [f.score for f in fold_results]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        
        return RepeatResult(
            repeat_id=repeat_id,
            fold_results=fold_results,
            mean_score=mean_score,
            std_score=std_score,
        )
    
    def _resolve_target_column(self, model: BiasModel, target_col: str) -> str:
        """Resolve target column with model-specific overrides"""
        if model.target_col_override is not None:
            return model.target_col_override
        
        if model.task == "reg" and target_col == "gt_idx":
            return "ground_truth"
        
        return target_col
    
    def _aggregate_results(self, repeat_results: List[RepeatResult]) -> tuple[float, float, int]:
        """Aggregate results across repeats"""
        repeat_means = [r.mean_score for r in repeat_results]
        overall_mean = float(np.mean(repeat_means))
        overall_std = float(np.std(repeat_means))
        total_count = repeat_results[0].total_instances if repeat_results else 0
        
        return overall_mean, overall_std, total_count
    
    def _log_results(self, result: EvaluationResult):
        """Log evaluation results"""
        from ezcolorlog import root_logger as logger
        
        logger.info(
            f"[{result.model_name.upper()}] "
            f"Overall {result.metric_name.upper()}: "
            f"{result.overall_mean:.2%} ± {result.overall_std:.2%} "
            f"(n_splits={self.config.n_splits}, repeats={self.config.repeats})"
        )
        
        if self.config.repeats == 1:
            fold_scores = [f.score for f in result.repeat_results[0].fold_results]
            logger.info(
                f"[{result.model_name.upper()}] "
                f"Fold {result.metric_name.upper()}s: "
                f"{[f'{s:.2%}' for s in fold_scores]}"
            )
        else:
            repeat_scores = [r.mean_score for r in result.repeat_results]
            logger.info(
                f"[{result.model_name.upper()}] "
                f"Repeat {result.metric_name.upper()}s: "
                f"{[f'{s:.2%}' for s in repeat_scores]}"
            )
```

### 3. Model-Specific Evaluators

**File**: `src/TsT/core/evaluators.py` (update from Phase 1)

```python
from .cross_validation import FoldEvaluator, PostProcessor
from .results import FoldResult, EvaluationResult

class RandomForestFoldEvaluator(FoldEvaluator):
    """RF-specific fold evaluator"""
    
    def evaluate_fold(
        self,
        model: FeatureBasedBiasModel,  # Feature-based model
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate RF on a single fold"""
        # Feature engineering
        model.fit_feature_maps(train_df)
        train_features = model.add_features(train_df)
        test_features = model.add_features(test_df)
        
        # Prepare data
        X_tr, X_te = train_features[model.feature_cols], test_features[model.feature_cols]
        encode_categoricals(X_tr, X_te)
        y_tr, y_te = train_features[target_col], test_features[target_col]
        
        # Train and evaluate
        estimator = _make_estimator(model.task, seed)
        estimator.fit(X_tr, y_tr)
        score = _score(estimator, X_te, y_te, model.metric)
        
        return FoldResult(
            fold_id=fold_id,
            score=score,
            fold_size=len(test_df),
            metadata={"estimator_params": estimator.get_params()},
        )

class RandomForestPostProcessor(PostProcessor):
    """Generate feature importances for RF models"""
    
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
        encode_categoricals(X_full, X_full.copy())
        y_full = full_df[target_col]
        
        estimator = _make_estimator(model.task, evaluation_result.config.random_state)
        estimator.fit(X_full, y_full)
        
        feature_importances = pd.DataFrame({
            "feature": model.feature_cols,
            "importance": estimator.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        
        # Update result
        evaluation_result.feature_importances = feature_importances
        return evaluation_result

class LLMFoldEvaluator(FoldEvaluator):
    """LLM-specific fold evaluator"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
        self.trainable_predictor = None
    
    def evaluate_fold(
        self,
        model: BiasModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate LLM on a single fold"""
        # Initialize predictor if needed
        if self.trainable_predictor is None:
            from ..llm.trainable.predictor import create_trainable_predictor
            self.trainable_predictor = create_trainable_predictor(self.llm_config)
        
        # Convert data
        from ..llm.data.conversion import convert_to_tst_format, convert_to_test_instances
        train_data = convert_to_tst_format(train_df, target_col, model.format)
        test_instances = convert_to_test_instances(test_df, target_col)
        
        # Train and evaluate
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainable_predictor.train(train_data, Path(temp_dir))
            predictions = self.trainable_predictor.predict(test_instances)
            score = self._calculate_accuracy(predictions, test_instances, model.format)
        
        return FoldResult(
            fold_id=fold_id,
            score=score,
            fold_size=len(test_df),
            metadata={
                "training_size": len(train_data),
                "model_name": self.llm_config["model_name"],
            },
        )

class LLMPostProcessor(PostProcessor):
    """Generate LLM-specific metadata"""
    
    def __init__(self, llm_config: Dict, zero_shot_baseline: float = None):
        self.llm_config = llm_config
        self.zero_shot_baseline = zero_shot_baseline
    
    def process_results(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Add LLM-specific metadata"""
        # Calculate improvement over zero-shot
        improvement = (evaluation_result.overall_mean - self.zero_shot_baseline 
                      if self.zero_shot_baseline else 0.0)
        
        # Mock feature importances for compatibility
        feature_importances = pd.DataFrame({
            "feature": ["llm_finetuning", "zero_shot_baseline", "improvement"],
            "importance": [evaluation_result.overall_mean, self.zero_shot_baseline or 0.0, improvement]
        })
        
        evaluation_result.feature_importances = feature_importances
        evaluation_result.model_metadata = {
            "zero_shot_baseline": self.zero_shot_baseline,
            "improvement": improvement,
            "llm_config": self.llm_config,
        }
        
        return evaluation_result
```

### 4. Updated Main Evaluation Function

**File**: `src/TsT/evaluation.py` (major refactor)

```python
from .core.cross_validation import UnifiedCrossValidator, CrossValidationConfig
from .core.evaluators import (
    RandomForestFoldEvaluator, RandomForestPostProcessor,
    LLMFoldEvaluator, LLMPostProcessor
)
from .core.results import EvaluationResult

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
    Unified evaluation function for all model types.
    Now uses the unified cross-validation framework.
    """
    # Filter models if specified
    if question_types is not None:
        models = [m for m in models if m.name in question_types]
    
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
        zero_shot_baseline = _get_zero_shot_baseline(models[0], df_full, target_col, llm_config)
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
    summary = pd.DataFrame(summary_data).sort_values("Score", ascending=False)
    
    # Calculate and log overall statistics
    _log_overall_statistics(summary)
    
    return summary

def _get_zero_shot_baseline(model: BiasModel, df: pd.DataFrame, target_col: str, llm_config: Dict) -> float:
    """Get zero-shot baseline for LLM evaluation"""
    # Implementation for zero-shot evaluation
    pass

def _create_error_result(model: BiasModel, error_msg: str) -> EvaluationResult:
    """Create error result for failed evaluations"""
    pass

def _log_overall_statistics(summary: pd.DataFrame):
    """Log overall evaluation statistics"""
    pass
```

## Success Criteria

1. **Unified Framework**: Single cross-validation engine handles all model types
2. **Consistent Reporting**: Identical progress tracking and result formatting
3. **Extensible**: Easy to add new model types with minimal boilerplate
4. **Maintainable**: Single source of truth for CV logic
5. **Rich Results**: Detailed result objects with fold-level metadata
6. **Backward Compatible**: Existing RF evaluation produces identical results

## Breaking Changes

- `evaluate_bias_model()` and `evaluate_bias_model_llm()` functions are removed
- Result format changes from simple tuple to rich `EvaluationResult` objects
- Progress bar formatting may change slightly

## Migration Strategy

1. **Deprecation Warnings**: Add warnings to old functions
2. **Adapter Layer**: Create compatibility wrappers if needed
3. **Documentation**: Update examples to use new API
4. **Testing**: Ensure identical results on existing benchmarks

## Estimated Effort

**3-4 days** - This is primarily a refactoring exercise that consolidates existing functionality into a cleaner architecture.
