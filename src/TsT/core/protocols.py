"""
All protocols for bias detection models and evaluators.

This module defines the complete interface hierarchy for the TsT evaluation framework:
- BiasModel: Generic interface for any bias detection model
- FeatureBasedBiasModel: Specific interface for traditional ML models with feature engineering
- ModelEvaluator: Abstract base for evaluation strategies
"""

from typing import Protocol, Optional, List, runtime_checkable, Literal, Dict, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from TsT.utils import weighted_mean_std


# Type aliases for better ergonomics - define once, use everywhere
QAFormat = Literal["mc", "num", "oe"]  # multiple choice, numerical, or open-ended
Task = Literal["clf", "reg", "oe"]  # classification, regression, or open-ended
Metric = Literal["acc", "mra"]  # accuracy or mean relative accuracy

########################################################
# Results
########################################################


@dataclass
class FoldResult:
    """Result from a single CV fold"""

    fold_id: int
    score: float
    train_size: int  # dont need train_idx
    test_idx: List[int]
    metric: Metric
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def test_size(self) -> int:
        return len(self.test_idx)


@dataclass
class RepeatResult:
    """Result from a single repeat (collection of folds)"""

    repeat_id: int
    fold_results: List[FoldResult]
    mean_score: float
    std_score: float

    # idx_to_scores

    @property
    def total_instances(self) -> int:
        return sum(f.test_size for f in self.fold_results)

    @property
    def num_folds(self) -> int:
        return len(self.fold_results)

    @classmethod
    def from_fold_results(cls, repeat_id: int, fold_results: List[FoldResult]) -> "RepeatResult":
        """Create RepeatResult from fold results with calculated statistics"""

        scores = [f.score for f in fold_results]
        counts = [f.test_size for f in fold_results]

        if scores and counts:
            mean_score, std_score = weighted_mean_std(scores, counts)
            # num_correct = np.round(mean_score @ counts)
        else:
            mean_score = 0.0
            std_score = 0.0

        return cls(
            repeat_id=repeat_id,
            fold_results=fold_results,
            mean_score=mean_score,
            std_score=std_score,
        )


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
    count: int
    repeats: int
    total_count: int

    # Model-specific metadata
    zero_shot_baseline: float = 0.0
    feature_importances: Optional[pd.DataFrame] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fold_scores_per_repeat(self) -> np.ndarray:
        """Get scores per repeat. Shape: (#repeats, #folds)"""
        return np.array([[fold.score for fold in repeat.fold_results] for repeat in self.repeat_results])

    @property
    def repeat_scores(self) -> np.ndarray:
        """Get scores per repeat. Shape: (#repeats,)"""
        return np.array([repeat.mean_score for repeat in self.repeat_results])

    @classmethod
    def from_repeat_results(
        cls,
        model_name: str,
        model_format: str,
        metric_name: str,
        repeat_results: List[RepeatResult],
        zero_shot_baseline: float = 0.0,
        feature_importances: Optional[pd.DataFrame] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationResult":
        """Create EvaluationResult from repeat results with calculated statistics"""
        # repeat_means = [r.mean_score for r in repeat_results]

        assert len(repeat_results) > 0, "Must have at least one repeat"
        assert all(len(repeat.fold_results) > 0 for repeat in repeat_results), "Each repeat must have at least one fold"

        flat_scores = [fold.score for repeat in repeat_results for fold in repeat.fold_results]
        flat_counts = [fold.test_size for repeat in repeat_results for fold in repeat.fold_results]

        # TODO: not doing flat scores
        if flat_scores and flat_counts:
            overall_mean, overall_std = weighted_mean_std(flat_scores, flat_counts)
        else:
            overall_mean = 0.0
            overall_std = 0.0

        total_count = sum(flat_counts)
        repeats = len(repeat_results)
        count = total_count // repeats
        assert count == repeat_results[0].total_instances, (
            f"Overall count {count} does not match repeat {repeat_results[0].repeat_id} total instances {repeat_results[0].total_instances}"
        )

        return cls(
            model_name=model_name,
            model_format=model_format,
            metric_name=metric_name,
            repeat_results=repeat_results,
            zero_shot_baseline=zero_shot_baseline,
            overall_mean=overall_mean,
            overall_std=overall_std,
            count=count,
            repeats=repeats,
            total_count=total_count,
            feature_importances=feature_importances,
            model_metadata=model_metadata or {},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create EvaluationResult from dictionary (e.g., loaded from JSON)"""
        # Convert nested dictionaries back to objects
        repeat_results = []
        for repeat_data in data["repeat_results"]:
            fold_results = []
            for fold_data in repeat_data["fold_results"]:
                fold_result = FoldResult(
                    fold_id=int(fold_data["fold_id"]),  # Ensure Python int
                    score=float(fold_data["score"]),  # Ensure Python float
                    train_size=int(fold_data["train_size"]),  # Ensure Python int
                    test_idx=[int(x) for x in fold_data["test_idx"]],  # Convert to list of Python ints
                    metric=fold_data["metric"],
                    metadata=fold_data.get("metadata", {}),
                )
                fold_results.append(fold_result)

            repeat_result = RepeatResult(
                repeat_id=int(repeat_data["repeat_id"]),  # Ensure Python int
                fold_results=fold_results,
                mean_score=float(repeat_data["mean_score"]),  # Ensure Python float
                std_score=float(repeat_data["std_score"]),  # Ensure Python float
            )
            repeat_results.append(repeat_result)

        # Handle feature_importances if present
        feature_importances = data.get("feature_importances")
        if feature_importances is not None:
            feature_importances = pd.DataFrame(feature_importances)

        return cls(
            model_name=data["model_name"],
            model_format=data["model_format"],
            metric_name=data["metric_name"],
            repeat_results=repeat_results,
            overall_mean=float(data["overall_mean"]),  # Ensure Python float
            overall_std=float(data["overall_std"]),  # Ensure Python float
            count=int(data["count"]),  # Ensure Python int
            repeats=int(data["repeats"]),  # Ensure Python int
            total_count=int(data["total_count"]),  # Ensure Python int
            zero_shot_baseline=float(data.get("zero_shot_baseline", 0.0)),  # Ensure Python float
            feature_importances=feature_importances,
            model_metadata=data.get("model_metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationResult to dict with proper type handling"""
        import dataclasses

        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert to dictionary and handle numpy types
        result_dict = dataclasses.asdict(self)
        result_dict = convert_numpy_types(result_dict)
        return result_dict

    def to_json(self) -> str:
        """Convert EvaluationResult to JSON string with proper type handling"""
        import json

        return json.dumps(self.to_dict())

    def save_to_file(self, filepath: str) -> None:
        """Save EvaluationResult to a JSON file"""
        import json
        from pathlib import Path

        result_dict = self.to_dict()

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "EvaluationResult":
        """Load EvaluationResult from a JSON file"""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


########################################################
# Models
########################################################


@runtime_checkable
class BiasModel(Protocol):
    """Base protocol for any bias detection model (RF, LLM, etc.)"""

    name: str  # e.g., "count_2d", ...
    benchmark_name: str  # e.g., "cvb", "vsi"
    format: QAFormat
    target_col_override: Optional[str] = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to relevant rows for this model"""
        ...

    @property
    def task(self) -> Task:
        """Classification or regression task"""
        ...

    @property
    def metric(self) -> Metric:
        """Accuracy or mean relative accuracy"""
        ...

    def __str__(self) -> str:
        return f"{self.name} ({self.format})"


@runtime_checkable
class FeatureBasedBiasModel(BiasModel, Protocol):
    """
    Protocol for feature-based bias detection models (e.g., Random Forest).

    Extends BiasModel with feature engineering capabilities required for
    traditional ML approaches. This is what was previously called 'QType'.
    """

    feature_cols: List[str]

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect any statistics derived from *train* only (for leakage‑free CV)."""
        ...

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a **copy** of `df` with question‑specific feature columns added."""
        ...

    @property
    def task(self) -> Task:
        match self.format:
            case "mc":
                return "clf"
            case "num":
                return "reg"
            case "oe":
                raise ValueError("Open-ended tasks are not supported for feature-based models")
            case _:
                raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self) -> Metric:
        match self.format:
            case "mc":
                return "acc"
            case "num":
                return "mra"
            case "oe":
                raise ValueError("Open-ended tasks are not supported for feature-based models")
            case _:
                raise ValueError(f"Unknown format: {self.format}")


@runtime_checkable
class QuestionAnswerBiasModel(BiasModel, Protocol):
    """
    Protocol for question-answer based bias detection models (e.g., LLMs).

    Unlike FeatureBasedBiasModel, these models work directly with
    question-answer pairs without feature engineering.
    """

    def prepare_instances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional preprocessing of data before evaluation"""
        return df

    @property
    def task(self) -> Task:
        match self.format:
            case "mc":
                return "clf"
            case "num":
                return "reg"
            case "oe":
                return "oe"
            case _:
                raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self) -> Metric:
        match self.format:
            case "mc":
                return "acc"
            case "num":
                return "mra"
            case "oe":
                raise NotImplementedError("TODO")
            case _:
                raise ValueError(f"Unknown format: {self.format}")


class ModelEvaluator(ABC):
    """Abstract base for model evaluation strategies"""

    @abstractmethod
    def train_and_evaluate_fold(
        self,
        model: BiasModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate a single fold and return score"""
        pass

    def process_results(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Default results processor that simply returns unmodified evaluation result."""
        return evaluation_result
