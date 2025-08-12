"""
Abstract interfaces for LLM predictors.

This module defines the base interfaces that all LLM predictors must implement,
following DataEnvGym patterns for clean abstraction and composability.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..data.models import TestInstance, LLMPredictionResult


class LLMPredictorInterface(ABC):
    """Abstract interface for LLM predictors"""

    @abstractmethod
    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """
        Generate predictions for test instances.

        Args:
            instances: List of test instances to predict on

        Returns:
            List of prediction results in the same order as input instances
        """
        pass

    @abstractmethod
    def load_adapter(self, adapter_path: str) -> None:
        """
        Load a LoRA adapter for fine-tuned inference.

        Args:
            adapter_path: Path to the LoRA adapter directory
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset model state and free GPU memory.

        This is critical for k-fold training where we need to free memory
        between training and inference operations.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the predictor is loaded and ready for inference"""
        pass

    @property
    @abstractmethod
    def current_adapter_path(self) -> Optional[str]:
        """Get the path of the currently loaded adapter, if any"""
        pass


class BaseLLMPredictor(LLMPredictorInterface):
    """
    Base implementation with common functionality for LLM predictors.

    This provides default implementations and common utilities that
    concrete predictors can inherit and extend.
    """

    def __init__(self):
        self._current_adapter_path: Optional[str] = None
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if the predictor is loaded and ready for inference"""
        return self._is_loaded

    @property
    def current_adapter_path(self) -> Optional[str]:
        """Get the path of the currently loaded adapter, if any"""
        return self._current_adapter_path

    def _set_adapter_path(self, adapter_path: str) -> None:
        """Set the current adapter path (for use by subclasses)"""
        self._current_adapter_path = adapter_path

    def _set_loaded(self, loaded: bool) -> None:
        """Set the loaded state (for use by subclasses)"""
        self._is_loaded = loaded

    def _validate_instances(self, instances: List[TestInstance]) -> None:
        """Validate input instances before prediction"""
        if not instances:
            raise ValueError("No instances provided for prediction")

        if not all(isinstance(inst, TestInstance) for inst in instances):
            raise TypeError("All instances must be TestInstance objects")

        # Check for duplicate instance IDs
        instance_ids = [inst.instance_id for inst in instances]
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("Duplicate instance IDs found")

    def _create_prediction_result(
        self,
        instance: TestInstance,
        prediction: str,
        confidence: Optional[float] = None,
        raw_output: Optional[str] = None,
    ) -> LLMPredictionResult:
        """Helper to create prediction result objects"""
        return LLMPredictionResult(
            instance_id=instance.instance_id,
            prediction=prediction.strip(),
            confidence=confidence,
            raw_output=raw_output,
        )
