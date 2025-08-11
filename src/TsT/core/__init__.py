"""
Core abstractions for the TsT evaluation framework.

This module provides base protocols and interfaces that allow different
model types (Random Forest, LLM, etc.) to work with a unified evaluation system.
"""

from .protocols import BiasModel, FeatureBasedBiasModel, ModelEvaluator
from .cross_validation import (
    UnifiedCrossValidator,
    CrossValidationConfig,
    FoldEvaluator,
    PostProcessor,
)
from .results import FoldResult, RepeatResult, EvaluationResult
from .evaluators import (
    RandomForestEvaluator,
    RandomForestFoldEvaluator,
    RandomForestPostProcessor,
    LLMFoldEvaluator,
    LLMPostProcessor,
)

__all__ = [
    # Protocol interfaces
    "BiasModel",
    "FeatureBasedBiasModel",
    "ModelEvaluator",
    # Unified evaluation framework
    "UnifiedCrossValidator",
    "CrossValidationConfig",
    "FoldEvaluator",
    "PostProcessor",
    "FoldResult",
    "RepeatResult",
    "EvaluationResult",
    # Model-specific evaluators and post-processors
    "RandomForestEvaluator",
    "RandomForestFoldEvaluator",
    "RandomForestPostProcessor",
    "LLMFoldEvaluator",
    "LLMPostProcessor",
]
