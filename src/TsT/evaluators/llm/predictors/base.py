"""
Interface for TsT-LLM predictors.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..data.models import LLMPredictionResult, TestInstance


class LLMPredictorInterface(ABC):
    """Generates predictions with a base model, optionally with a LoRA adapter loaded."""

    @abstractmethod
    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """Predictions for ``instances``, in the same order."""

    @abstractmethod
    def load_adapter(self, adapter_path: str) -> None:
        """Use the LoRA adapter at ``adapter_path`` for subsequent predictions."""

    @abstractmethod
    def reset(self) -> None:
        """Unload the model and free GPU memory (before LoRA training)."""

    @abstractmethod
    def ensure_loaded(self) -> None:
        """Load the base model if it is not loaded."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the base model is loaded."""

    @property
    @abstractmethod
    def current_adapter_path(self) -> Optional[str]:
        """The loaded adapter, if any."""


def validate_instances(instances: List[TestInstance]) -> None:
    if not instances:
        raise ValueError("No instances provided for prediction")
    if not all(isinstance(inst, TestInstance) for inst in instances):
        raise TypeError("All instances must be TestInstance objects")
    ids = [inst.instance_id for inst in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate instance IDs found")
