"""
Model evaluators for different bias detection approaches.

This module contains evaluator classes that implement the actual evaluation
logic for different model types while working with the unified cross-validation framework.
"""

from .rf import RandomForestEvaluator
from .llm import LLMEvaluator

__all__ = [
    "RandomForestEvaluator",
    "LLMEvaluator",
]
