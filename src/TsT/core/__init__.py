"""
Core abstractions for the TsT evaluation framework.

This module provides base protocols and interfaces that allow different
model types (Random Forest, LLM, etc.) to work with a unified evaluation system.
"""

from .protocols import BiasModel, FeatureBasedBiasModel, QuestionAnswerBiasModel, ModelEvaluator
from .protocols import FoldResult, RepeatResult, EvaluationResult
from .cross_validation import (
    UnifiedCrossValidator,
    CrossValidationConfig,
)
from .benchmark import Benchmark, BenchmarkRegistry
from .qa_models import GlobalBenchmarkQAModel


__all__ = [
    # Protocol interfaces
    "BiasModel",
    "FeatureBasedBiasModel",
    "QuestionAnswerBiasModel",
    "ModelEvaluator",
    # Benchmark system
    "Benchmark",
    "BenchmarkRegistry",
    "GlobalBenchmarkQAModel",
    # Unified evaluation framework
    "UnifiedCrossValidator",
    "CrossValidationConfig",
    "FoldResult",
    "RepeatResult",
    "EvaluationResult",
]
