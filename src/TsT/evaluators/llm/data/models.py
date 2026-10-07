"""
Records passed between the TsT-LLM evaluator, trainer and predictor.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingDatum:
    """One fine-tuning example: a blind prompt and its answer."""

    instruction: str
    response: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestInstance:
    """One held-out question to score."""

    __test__ = False  # not a pytest test class

    instance_id: str
    instruction: str
    ground_truth: str
    options: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMPredictionResult:
    """A prediction for one TestInstance."""

    instance_id: str
    prediction: str
    # Probability of the gold option (MC with log-probability scoring); None when the gold
    # letter is not among the returned top log-probabilities.
    confidence: Optional[float] = None
    # Probabilities over the option letters found in the top log-probabilities, normalized to 1.
    option_probs: Optional[Dict[str, float]] = None
    raw_output: Optional[str] = None


@dataclass
class LoRAAdapterInfo:
    """A trained LoRA adapter."""

    adapter_path: Path
    training_size: int
    model_name: str
    training_config: Dict[str, Any] = field(default_factory=dict)
