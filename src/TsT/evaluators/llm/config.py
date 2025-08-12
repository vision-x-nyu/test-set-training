"""
Typed configuration for LLM runs (predictor + trainer fan-out).

Provides a single dataclass that can be converted into the specific
predictor and trainer configs used by the system.
"""

from dataclasses import dataclass

from .predictors.vllm import VLLMPredictorConfig
from .trainers.llamafactory import LlamaFactoryConfig


@dataclass
class LLMRunConfig:
    """Typed configuration that covers both inference and training.

    This aggregates the common knobs we currently expose for vLLM inference
    and LlamaFactory LoRA fine-tuning and can be fanned out to each.
    """

    # Shared core
    model_name: str = "google/gemma-2-2b-it"
    max_seq_length: int = 512

    # Inference (predictor)
    eval_batch_size: int = 32
    temperature: float = 0.0
    max_tokens: int = 10
    apply_chat_template: bool = False

    # Training (trainer)
    learning_rate: float = 2e-4
    train_batch_size: int = 32
    num_epochs: int = 5
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    template: str = "gemma"

    def to_predictor_config(self) -> VLLMPredictorConfig:
        """Create a predictor config from this run config."""
        return VLLMPredictorConfig(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            batch_size=self.eval_batch_size,
            apply_chat_template=self.apply_chat_template,
        )

    def to_trainer_config(self) -> LlamaFactoryConfig:
        """Create a trainer config from this run config."""
        return LlamaFactoryConfig(
            model_name=self.model_name,
            template=self.template,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            max_seq_length=self.max_seq_length,
        )
