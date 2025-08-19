"""
Core abstractions for the TsT evaluation framework.

This module provides base protocols and interfaces that allow different
model types (Random Forest, LLM, etc.) to work with a unified evaluation system.
"""

from .protocols import BiasModel, FeatureBasedBiasModel, QuestionAnswerBiasModel, ModelEvaluator
from .cross_validation import (
    UnifiedCrossValidator,
    CrossValidationConfig,
)
from .results import FoldResult, RepeatResult, EvaluationResult
from .benchmark import Benchmark, BenchmarkRegistry
from .qa_models import SimpleBenchmarkQAModel


__all__ = [
    # Protocol interfaces
    "BiasModel",
    "FeatureBasedBiasModel",
    "QuestionAnswerBiasModel",
    "ModelEvaluator",
    # Benchmark system
    "Benchmark",
    "BenchmarkRegistry",
    "SimpleBenchmarkQAModel",
    # Unified evaluation framework
    "UnifiedCrossValidator",
    "CrossValidationConfig",
    "FoldResult",
    "RepeatResult",
    "EvaluationResult",
]
