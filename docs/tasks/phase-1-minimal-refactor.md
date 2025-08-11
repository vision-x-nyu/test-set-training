# Phase 1: Minimal Refactor for LLM Integration

## Background

The current TsT (Train-on-Test-Set) evaluation framework was designed around Random Forest models and feature engineering. We now need to add LLM-based bias detection as an alternative approach. However, the current architecture has significant issues:

1. **Code Duplication**: `evaluate_bias_model()` and `evaluate_bias_model_llm()` share ~80% of their code
2. **RF-Centric Protocol**: The `QType` protocol assumes feature-based models with methods like `fit_feature_maps()` and `add_features()`
3. **Hardcoded Logic**: Cross-validation, progress tracking, and statistics calculation are duplicated

## Objectives

Create minimal abstractions that allow clean LLM implementation without disrupting existing RF functionality.

## Implementation Plan

### 1. Create Base Abstractions

**File**: `src/TsT/core/protocols.py`

```python
from typing import Protocol, Literal, Optional
from abc import ABC, abstractmethod

class BiasModel(Protocol):
    """Base protocol for any bias detection model (RF, LLM, etc.)"""
    name: str
    format: Literal["mc", "num"]  # multiple choice or numerical
    target_col_override: Optional[str] = None
    
    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to relevant rows for this model"""
        ...
    
    @property
    def task(self) -> Literal["clf", "reg"]:
        """Classification or regression task"""
        ...
    
    @property  
    def metric(self) -> Literal["acc", "mra"]:
        """Accuracy or mean relative accuracy"""
        ...

class ModelEvaluator(ABC):
    """Abstract base for model evaluation strategies"""
    
    @abstractmethod
    def evaluate_fold(
        self, 
        model: BiasModel,
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        target_col: str,
        fold_num: int,
        seed: int
    ) -> float:
        """Evaluate a single fold and return score"""
        pass
```

### 2. Extend Existing QType

**File**: `src/TsT/protocols.py` (modify existing)

```python
# Make QType inherit from BiasModel to maintain backward compatibility
class QType(BiasModel, Protocol):
    feature_cols: List[str]
    
    def fit_feature_maps(self, train_df: pd.DataFrame) -> None: ...
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### 3. Create RF Evaluator

**File**: `src/TsT/core/evaluators.py`

```python
class RandomForestEvaluator(ModelEvaluator):
    """Evaluator for feature-based Random Forest models"""
    
    def evaluate_fold(
        self, 
        model: QType,  # Feature-based model
        train_df: pd.DataFrame,
        test_df: pd.DataFrame, 
        target_col: str,
        fold_num: int,
        seed: int
    ) -> float:
        # Extract existing RF evaluation logic
        model.fit_feature_maps(train_df)
        train_df = model.add_features(train_df)
        test_df = model.add_features(test_df)
        
        X_tr, X_te = train_df[model.feature_cols], test_df[model.feature_cols]
        encode_categoricals(X_tr, X_te)
        y_tr, y_te = train_df[target_col], test_df[target_col]
        
        est = _make_estimator(model.task, seed)
        est.fit(X_tr, y_tr)
        return _score(est, X_te, y_te, model.metric)
```

### 4. Extract Common Cross-Validation Logic

**File**: `src/TsT/core/cross_validation.py`

```python
def run_cross_validation(
    model: BiasModel,
    evaluator: ModelEvaluator,
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    repeats: int = 1,
    target_col: str = "ground_truth",
) -> tuple[float, float, int]:
    """
    Common cross-validation logic for any model type.
    Returns (mean_score, std_score, count)
    """
    qdf = model.select_rows(df)
    all_scores = []
    
    # Handle target column override logic
    if model.target_col_override is not None:
        target_col = model.target_col_override
    if model.task == "reg" and target_col == "gt_idx":
        target_col = "ground_truth"
    
    # Progress tracking
    repeat_pbar = tqdm(range(repeats), desc=f"[{model.name.upper()}] Repeats", disable=repeats == 1)
    
    for repeat in repeat_pbar:
        current_seed = random_state + repeat
        
        # Create appropriate splitter
        if model.task == "reg":
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf,)
        else:
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
            tr, te = qdf.iloc[tr_idx].copy(), qdf.iloc[te_idx].copy()
            
            # Delegate fold evaluation to the evaluator
            fold_score = evaluator.evaluate_fold(model, tr, te, target_col, fold, current_seed)
            scores.append(fold_score)
            
            fold_pbar.set_postfix({f"fold_{model.metric}": f"{np.mean(scores):.2%}"})
        
        all_scores.append(scores)
        if repeats > 1:
            repeat_pbar.set_postfix({f"avg_{model.metric}": f"{np.mean(scores):.2%}"})
    
    # Calculate statistics
    mean_scores = [np.mean(scores) for scores in all_scores]
    mean_acc = float(np.mean(mean_scores))
    std_acc = float(np.std(mean_scores))
    count = len(qdf)
    
    if verbose:
        logger.info(f"[{model.name.upper()}] Overall {model.metric.upper()}: {mean_acc:.2%} ± {std_acc:.2%}")
        if repeats == 1:
            logger.info(f"[{model.name.upper()}] Fold scores: {[f'{s:.2%}' for s in all_scores[0]]}")
    
    return mean_acc, std_acc, count
```

### 5. Update Main Evaluation Function

**File**: `src/TsT/evaluation.py` (modify existing)

```python
def run_evaluation(
    models: List[BiasModel],  # More generic type
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
    
    for model in models:
        try:
            if mode == "rf":
                evaluator = RandomForestEvaluator()
                mean_score, std_score, count = run_cross_validation(
                    model, evaluator, df_full, n_splits, random_state, verbose, repeats, target_col
                )
                # Generate feature importances for RF
                fi = _generate_feature_importances(model, df_full, target_col, random_state)
                
            elif mode == "llm":
                evaluator = LLMEvaluator(llm_config or {})
                mean_score, std_score, count = run_cross_validation(
                    model, evaluator, df_full, n_splits, random_state, verbose, repeats, target_col
                )
                # Mock feature importances for LLM
                fi = _generate_llm_feature_importances(mean_score)
            
            # Rest of result processing...
```

## Success Criteria

1. **No Duplication**: Common CV logic is extracted and reused
2. **Backward Compatibility**: Existing RF models work unchanged
3. **Clean Extension Point**: LLM evaluator can be implemented cleanly
4. **Type Safety**: Proper protocols for different model types
5. **Maintainability**: Single source of truth for CV logic

## Files to Create/Modify

- **Create**: `src/TsT/core/__init__.py`
- **Create**: `src/TsT/core/protocols.py`
- **Create**: `src/TsT/core/evaluators.py`
- **Create**: `src/TsT/core/cross_validation.py`
- **Modify**: `src/TsT/protocols.py`
- **Modify**: `src/TsT/evaluation.py`

## Testing Strategy

### Unit Tests (pytest)
1. **Cross-Validation Logic**: Test `run_cross_validation()` with mock evaluator
2. **RandomForestEvaluator**: Test `evaluate_fold()` with simple synthetic data
3. **Protocol Compliance**: Ensure existing models work with new `BiasModel` protocol
4. **Edge Cases**: Test error handling for invalid inputs

### Integration Tests
1. **Backward Compatibility**: Compare old vs new RF evaluation results
2. **End-to-End**: Test complete evaluation pipeline with Video-MME
3. **Error Handling**: Test graceful failure when evaluation fails

### Test Files to Create
- `tests/core/test_cross_validation.py` - Test unified CV logic
- `tests/core/test_evaluators.py` - Test RandomForestEvaluator 
- `tests/core/test_protocols.py` - Test protocol compliance
- `tests/test_evaluation_integration.py` - Integration tests

### Critical Test Cases
- CV with different splitters (StratifiedKFold vs KFold)
- Progress tracking and logging
- Feature importance generation
- Model selection and filtering
- Error propagation and handling

## Estimated Effort

**2-3 days** - This is a refactoring exercise that should be done carefully to maintain backward compatibility.
