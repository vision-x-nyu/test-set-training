"""
Core abstractions for the TsT evaluation framework.

This module provides base protocols and interfaces that allow different
model types (Random Forest, LLM, etc.) to work with a unified evaluation system.
"""

from .protocols import BiasModel, FeatureBasedBiasModel, ModelEvaluator
from .cross_validation import run_cross_validation
from .evaluators import RandomForestEvaluator

__all__ = [
    "BiasModel",
    "FeatureBasedBiasModel",
    "ModelEvaluator",
    "run_cross_validation",
    "RandomForestEvaluator",
]
