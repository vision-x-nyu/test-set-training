"""
Core abstractions for the TsT evaluation framework.

This module provides base protocols and interfaces that allow different
model types (Random Forest, LLM, etc.) to work with a unified evaluation system.
"""

from .protocols import BiasModel, FeatureBasedBiasModel, ModelEvaluator
from .cross_validation import (
    run_cross_validation,
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
    # Legacy interfaces
    "BiasModel",
    "FeatureBasedBiasModel",
    "ModelEvaluator",
    "run_cross_validation",
    "RandomForestEvaluator",
    # Unified evaluation framework
    "UnifiedCrossValidator",
    "CrossValidationConfig",
    "FoldEvaluator",
    "PostProcessor",
    "FoldResult",
    "RepeatResult",
    "EvaluationResult",
    # Model-specific evaluators and post-processors
    "RandomForestFoldEvaluator",
    "RandomForestPostProcessor",
    "LLMFoldEvaluator",
    "LLMPostProcessor",
]
