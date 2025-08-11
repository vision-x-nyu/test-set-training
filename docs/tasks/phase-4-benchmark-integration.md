# Phase 4: Benchmark-Specific Integration

## Background

After establishing unified evaluation framework (Phase 3), we now have clean abstractions for different model types. However, there's still benchmark-specific logic scattered throughout the codebase, particularly in the LLM data conversion functions.

Current issues:
1. **Hardcoded Conversions**: `_convert_to_blind_qa_format()` has benchmark-specific logic
2. **Inflexible Data Handling**: Each benchmark has different column names and formats
3. **Scattered Logic**: Benchmark knowledge is spread across evaluation and conversion code
4. **Hard to Extend**: Adding new benchmarks requires modifying core evaluation code

The goal is to move all benchmark-specific knowledge into the benchmark modules themselves, making the core evaluation framework truly generic.

## Objectives

1. **Encapsulate Benchmark Logic**: Each benchmark handles its own data formatting
2. **Generic Core Framework**: Evaluation code has no benchmark-specific knowledge
3. **Extensible Design**: New benchmarks can be added without touching core code
4. **Type Safety**: Proper interfaces for benchmark-specific functionality

## Implementation Plan

### 1. Benchmark Data Formatting Interface

**File**: `src/TsT/core/interfaces.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Protocol
import pandas as pd

class TextDataFormatter(Protocol):
    """Protocol for converting benchmark data to text format for LLMs"""
    
    def format_for_training(
        self, 
        df: pd.DataFrame, 
        target_col: str
    ) -> List[Dict[str, str]]:
        """
        Convert dataframe rows to training format.
        
        Returns:
            List of {"instruction": str, "response": str} dicts
        """
        ...
    
    def format_for_inference(
        self, 
        df: pd.DataFrame
    ) -> List[str]:
        """
        Convert dataframe rows to inference prompts.
        
        Returns:
            List of instruction strings
        """
        ...
    
    def get_ground_truth(
        self, 
        df: pd.DataFrame, 
        target_col: str
    ) -> List[str]:
        """
        Extract ground truth answers from dataframe.
        
        Returns:
            List of ground truth answer strings
        """
        ...

class BenchmarkInterface(Protocol):
    """Extended interface for benchmarks that support text formatting"""
    
    # Existing FeatureBasedBiasModel methods
    name: str
    format: Literal["mc", "num"]
    target_col_override: Optional[str]
    
    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame: ...
    
    @property
    def task(self) -> Literal["clf", "reg"]: ...
    
    @property
    def metric(self) -> Literal["acc", "mra"]: ...
    
    # New text formatting capability
    def get_text_formatter(self) -> TextDataFormatter:
        """Return text formatter for this benchmark"""
        ...
```

### 2. Generic Text Data Conversion

**File**: `src/TsT/llm/data/conversion.py`

```python
from typing import List, Dict
import pandas as pd
from ..core.interfaces import BenchmarkInterface, TextDataFormatter
from .models import TrainingDatum, TestInstance

def convert_to_training_data(
    benchmark: BenchmarkInterface,
    df: pd.DataFrame,
    target_col: str,
) -> List[TrainingDatum]:
    """
    Generic conversion to training data using benchmark's formatter.
    """
    formatter = benchmark.get_text_formatter()
    formatted_data = formatter.format_for_training(df, target_col)
    
    return [
        TrainingDatum(
            instruction=item["instruction"],
            response=item["response"],
            metadata={"benchmark": benchmark.name, "format": benchmark.format}
        )
        for item in formatted_data
    ]

def convert_to_test_instances(
    benchmark: BenchmarkInterface,
    df: pd.DataFrame,
    target_col: str,
) -> List[TestInstance]:
    """
    Generic conversion to test instances using benchmark's formatter.
    """
    formatter = benchmark.get_text_formatter()
    instructions = formatter.format_for_inference(df)
    ground_truths = formatter.get_ground_truth(df, target_col)
    
    return [
        TestInstance(
            instruction=instruction,
            instance_id=f"{benchmark.name}_{i}",
            ground_truth=ground_truth,
        )
        for i, (instruction, ground_truth) in enumerate(zip(instructions, ground_truths))
    ]

def calculate_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    benchmark_format: str,
) -> float:
    """
    Calculate accuracy based on benchmark format.
    """
    if benchmark_format == "mc":
        # Multiple choice: exact match (case insensitive)
        correct = sum(
            1 for pred, gt in zip(predictions, ground_truths)
            if pred.strip().upper() == gt.strip().upper()
        )
    else:
        # Numerical: exact string match
        correct = sum(
            1 for pred, gt in zip(predictions, ground_truths)
            if pred.strip() == gt.strip()
        )
    
    return correct / len(predictions) if predictions else 0.0
```

### 3. Video-MME Text Formatter

**File**: `src/TsT/benchmarks/video_mme/text_formatter.py`

```python
from typing import List, Dict
import pandas as pd
from ...core.interfaces import TextDataFormatter

class VideoMMETextFormatter(TextDataFormatter):
    """Video-MME specific text formatting for LLMs"""
    
    def format_for_training(
        self, 
        df: pd.DataFrame, 
        target_col: str
    ) -> List[Dict[str, str]]:
        """Convert Video-MME data to training format"""
        training_data = []
        
        for _, row in df.iterrows():
            # Extract question
            question = self._extract_question(row)
            
            # Format with multiple choice options
            instruction = self._format_instruction_with_choices(question, row["options"])
            
            # Get answer
            answer = self._format_answer(row, target_col)
            
            training_data.append({
                "instruction": instruction,
                "response": answer,
            })
        
        return training_data
    
    def format_for_inference(self, df: pd.DataFrame) -> List[str]:
        """Convert Video-MME data to inference prompts"""
        instructions = []
        
        for _, row in df.iterrows():
            question = self._extract_question(row)
            instruction = self._format_instruction_with_choices(question, row["options"])
            instructions.append(instruction)
        
        return instructions
    
    def get_ground_truth(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Extract ground truth answers"""
        return [self._format_answer(row, target_col) for _, row in df.iterrows()]
    
    def _extract_question(self, row: pd.Series) -> str:
        """Extract question text from Video-MME row"""
        # Video-MME uses 'question' column
        return row["question"]
    
    def _format_instruction_with_choices(self, question: str, options: List[str]) -> str:
        """Format question with multiple choice options"""
        # Format options as (A) option1 (B) option2 etc.
        choices_text = " ".join([
            f"({chr(65 + i)}) {option}" 
            for i, option in enumerate(options)
        ])
        
        post_prompt = "Answer with the option's letter from the given choices directly."
        return f"{question} Choices: {choices_text}\n{post_prompt}"
    
    def _format_answer(self, row: pd.Series, target_col: str) -> str:
        """Format answer based on target column"""
        if target_col == "gt_idx":
            # Convert index to letter (0->A, 1->B, etc.)
            return chr(65 + int(row[target_col]))
        else:
            # Use answer directly
            return str(row[target_col])
```

### 4. Update Video-MME Models

**File**: `src/TsT/benchmarks/video_mme/models.py` (modify existing)

```python
from ...core.interfaces import TextDataFormatter, BenchmarkInterface
from .text_formatter import VideoMMETextFormatter

class VideoMMEModel(BenchmarkInterface):  # Changed from FeatureBasedBiasModel
    name = "video_mme"
    format = "mc"
    feature_cols = FEATURE_COLS  # Keep for RF compatibility
    
    def __init__(self):
        # Existing initialization...
        self._text_formatter = VideoMMETextFormatter()
    
    # Existing FeatureBasedBiasModel methods remain unchanged...
    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def fit_feature_maps(self, train_df: pd.DataFrame) -> None: ...
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame: ...
    
    # New text formatting capability
    def get_text_formatter(self) -> TextDataFormatter:
        """Return Video-MME text formatter"""
        return self._text_formatter

class VideoMMEModelSubset(VideoMMEModel):
    """Inherits text formatting from parent"""
    pass

class VideoMMEModelSubsetCombo(VideoMMEModel):
    """Inherits text formatting from parent"""
    pass
```

### 5. Create Base Text Formatters

**File**: `src/TsT/benchmarks/base_formatters.py`

```python
from typing import List, Dict
import pandas as pd
from ..core.interfaces import TextDataFormatter

class MultipleChoiceFormatter(TextDataFormatter):
    """Generic formatter for multiple choice questions"""
    
    def __init__(
        self,
        question_col: str = "question",
        options_col: str = "options",
        answer_col: str = "answer",
        gt_idx_col: str = "gt_idx",
    ):
        self.question_col = question_col
        self.options_col = options_col
        self.answer_col = answer_col
        self.gt_idx_col = gt_idx_col
    
    def format_for_training(self, df: pd.DataFrame, target_col: str) -> List[Dict[str, str]]:
        """Generic MC training format"""
        training_data = []
        
        for _, row in df.iterrows():
            question = row[self.question_col]
            options = row[self.options_col]
            
            # Format with choices
            instruction = self._format_with_choices(question, options)
            answer = self._format_answer(row, target_col)
            
            training_data.append({
                "instruction": instruction,
                "response": answer,
            })
        
        return training_data
    
    def format_for_inference(self, df: pd.DataFrame) -> List[str]:
        """Generic MC inference format"""
        return [
            self._format_with_choices(row[self.question_col], row[self.options_col])
            for _, row in df.iterrows()
        ]
    
    def get_ground_truth(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Generic MC ground truth extraction"""
        return [self._format_answer(row, target_col) for _, row in df.iterrows()]
    
    def _format_with_choices(self, question: str, options: List[str]) -> str:
        """Format question with multiple choice options"""
        choices_text = " ".join([
            f"({chr(65 + i)}) {option}" 
            for i, option in enumerate(options)
        ])
        
        post_prompt = "Answer with the option's letter from the given choices directly."
        return f"{question} Choices: {choices_text}\n{post_prompt}"
    
    def _format_answer(self, row: pd.Series, target_col: str) -> str:
        """Format answer based on target column"""
        if target_col == self.gt_idx_col:
            return chr(65 + int(row[target_col]))
        else:
            return str(row[target_col])

class NumericalFormatter(TextDataFormatter):
    """Generic formatter for numerical questions"""
    
    def __init__(self, question_col: str = "question", answer_col: str = "ground_truth"):
        self.question_col = question_col
        self.answer_col = answer_col
    
    def format_for_training(self, df: pd.DataFrame, target_col: str) -> List[Dict[str, str]]:
        """Generic numerical training format"""
        return [
            {
                "instruction": row[self.question_col],
                "response": str(row[target_col]),
            }
            for _, row in df.iterrows()
        ]
    
    def format_for_inference(self, df: pd.DataFrame) -> List[str]:
        """Generic numerical inference format"""
        return [row[self.question_col] for _, row in df.iterrows()]
    
    def get_ground_truth(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Generic numerical ground truth extraction"""
        return [str(row[target_col]) for _, row in df.iterrows()]
```

### 6. Simplified LLM Evaluator

**File**: `src/TsT/core/evaluators.py` (update from Phase 3)

```python
class LLMFoldEvaluator(FoldEvaluator):
    """LLM-specific fold evaluator - now benchmark agnostic"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
        self.trainable_predictor = None
    
    def evaluate_fold(
        self,
        model: BenchmarkInterface,  # Now uses benchmark interface
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate LLM on a single fold - benchmark agnostic"""
        # Initialize predictor if needed
        if self.trainable_predictor is None:
            from ..llm.trainable.predictor import create_trainable_predictor
            self.trainable_predictor = create_trainable_predictor(self.llm_config)
        
        # Convert data using benchmark's formatter
        from ..llm.data.conversion import (
            convert_to_training_data, 
            convert_to_test_instances,
            calculate_accuracy
        )
        
        train_data = convert_to_training_data(model, train_df, target_col)
        test_instances = convert_to_test_instances(model, test_df, target_col)
        
        # Train and evaluate
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainable_predictor.train(train_data, Path(temp_dir))
            predictions = self.trainable_predictor.predict(test_instances)
            
            # Extract predictions and calculate accuracy
            pred_strings = [p.prediction for p in predictions]
            gt_strings = [t.ground_truth for t in test_instances]
            score = calculate_accuracy(pred_strings, gt_strings, model.format)
        
        return FoldResult(
            fold_id=fold_id,
            score=score,
            fold_size=len(test_df),
            metadata={
                "training_size": len(train_data),
                "model_name": self.llm_config["model_name"],
                "benchmark": model.name,
            },
        )
```

### 7. Migration Guide for Existing Benchmarks

**File**: `src/TsT/benchmarks/migration_guide.md`

```markdown
# Migrating Benchmarks to Text Formatting

## Quick Migration

For simple multiple choice benchmarks:

```python
from ..base_formatters import MultipleChoiceFormatter
from ...core.interfaces import BenchmarkInterface

class YourBenchmarkModel(BenchmarkInterface):
    def __init__(self):
        # Existing initialization...
        self._text_formatter = MultipleChoiceFormatter(
            question_col="your_question_column",
            options_col="your_options_column", 
            answer_col="your_answer_column",
            gt_idx_col="your_gt_idx_column",
        )
    
    def get_text_formatter(self):
        return self._text_formatter
```

## Custom Formatting

For benchmarks with special formatting needs:

```python
from ...core.interfaces import TextDataFormatter

class CustomFormatter(TextDataFormatter):
    def format_for_training(self, df, target_col):
        # Your custom logic here
        pass
    
    # Implement other methods...
```

## Testing

Ensure your formatter works:

```python
# Test with sample data
formatter = YourFormatter()
train_data = formatter.format_for_training(sample_df, "target_col")
assert all("instruction" in item and "response" in item for item in train_data)
```
```

### 8. Update Other Benchmarks

Apply similar patterns to:
- **VSI-Bench**: Create `VSITextFormatter` 
- **CV-Bench**: Create `CVBenchTextFormatter`
- **MMMU**: Create `MMUTextFormatter`

Each follows the same pattern:
1. Create benchmark-specific formatter
2. Add `get_text_formatter()` method to model class
3. Handle any special formatting requirements

## Success Criteria

1. **No Hardcoded Logic**: Core evaluation code has no benchmark-specific knowledge
2. **Extensible**: New benchmarks can be added without touching core code
3. **Consistent Interface**: All benchmarks implement same text formatting protocol
4. **Backward Compatible**: Existing RF evaluation continues to work
5. **Maintainable**: Benchmark logic is encapsulated in benchmark modules

## Breaking Changes

- Models must implement `BenchmarkInterface` instead of just `FeatureBasedBiasModel`
- LLM evaluation requires models to provide text formatters
- `_convert_to_blind_qa_format()` function is removed

## Migration Strategy

1. **Create Base Formatters**: Start with generic MC and numerical formatters
2. **Migrate One Benchmark**: Start with Video-MME as proof of concept
3. **Update Core Code**: Remove hardcoded conversion logic
4. **Migrate Remaining**: Apply pattern to other benchmarks
5. **Clean Up**: Remove old conversion functions

## Estimated Effort

**1 week** - This is mostly organizational work, moving existing logic into better locations.

## Dependencies

No new dependencies required - this is pure refactoring.

## Testing Strategy

1. **Unit Tests**: Test each formatter independently
2. **Integration Tests**: Ensure LLM evaluation works with new formatters
3. **Regression Tests**: Verify RF evaluation unchanged
4. **Cross-Benchmark Tests**: Ensure consistent behavior across benchmarks
