"""
Pydantic data models for TsT LLM operations.

This module defines type-safe data models for LLM training and inference,
following DataEnvGym patterns for robustness and maintainability.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, Any
from pathlib import Path


class TrainingDatum(BaseModel):
    """Training data for TsT LLM fine-tuning"""

    instruction: str = Field(..., description="The instruction/question for the model")
    response: str = Field(..., description="The expected response/answer")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for tracking")


class TestInstance(BaseModel):
    """Test instance for TsT LLM inference"""

    instance_id: str = Field(..., description="Unique identifier for this instance")
    instruction: str = Field(..., description="The instruction/question for the model")
    ground_truth: str = Field(..., description="The correct answer for evaluation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for tracking")


class LLMPredictionResult(BaseModel):
    """Result from LLM prediction"""

    instance_id: str = Field(..., description="ID matching the test instance")
    prediction: str = Field(..., description="The model's prediction")
    confidence: Optional[float] = Field(None, description="Optional confidence score")
    raw_output: Optional[str] = Field(None, description="Full raw output from the model")


class LoRAAdapterInfo(BaseModel):
    """Information about a trained LoRA adapter"""

    fold_id: int = Field(..., description="Which k-fold this adapter was trained on")
    adapter_path: Path = Field(..., description="Path to the saved adapter")
    training_size: int = Field(..., description="Number of training examples used")
    model_name: str = Field(..., description="Base model name that was fine-tuned")
    training_config: Dict[str, Any] = Field(..., description="Training configuration used")

    # Allow non-pydantic types like pathlib.Path (kept for compatibility)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMEvaluationMetrics(BaseModel):
    """Metrics from LLM evaluation"""

    accuracy: float = Field(..., description="Accuracy score")
    total_instances: int = Field(..., description="Total number of test instances")
    correct_predictions: int = Field(..., description="Number of correct predictions")
    evaluation_time_seconds: float = Field(..., description="Time taken for evaluation")


class TrainingProgress(BaseModel):
    """Progress information during training"""

    epoch: int = Field(..., description="Current epoch")
    step: int = Field(..., description="Current step within epoch")
    loss: float = Field(..., description="Current training loss")
    learning_rate: float = Field(..., description="Current learning rate")
    timestamp: str = Field(..., description="ISO timestamp of this progress update")


class LLMConfig(BaseModel):
    """Base configuration for LLM operations"""

    model_name: str = Field(..., description="Model name/path")
    max_seq_length: int = Field(512, description="Maximum sequence length")
    temperature: float = Field(0.0, description="Sampling temperature")
    max_tokens: int = Field(10, description="Maximum tokens to generate")

    # Allow additional config fields to pass through
    model_config = ConfigDict(extra="allow")
