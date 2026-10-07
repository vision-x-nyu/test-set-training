"""
Model evaluators for different bias detection approaches.

This module contains evaluator classes that implement the per-fold training and
scoring logic used by the unified cross-validation framework.
"""

from .rf import RandomForestEvaluator

__all__ = [
    "RandomForestEvaluator",
]
