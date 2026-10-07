"""
Core abstractions for the TsT evaluation framework.

This module provides base protocols and interfaces that allow different
model types to work with a unified k-fold evaluation system.
"""

from .protocols import BiasModel, FeatureBasedBiasModel, QuestionAnswerBiasModel, ModelEvaluator, IBPStrategy
from .protocols import FoldResult, RepeatResult, EvaluationResult
from .cross_validation import (
    UnifiedCrossValidator,
    CrossValidationConfig,
)
from .benchmark import Benchmark, BenchmarkRegistry


__all__ = [
    # Protocol interfaces
    "BiasModel",
    "FeatureBasedBiasModel",
    "QuestionAnswerBiasModel",
    "IBPStrategy",
    "ModelEvaluator",
    # Benchmark system
    "Benchmark",
    "BenchmarkRegistry",
    # Unified evaluation framework
    "UnifiedCrossValidator",
    "CrossValidationConfig",
    "FoldResult",
    "RepeatResult",
    "EvaluationResult",
]
