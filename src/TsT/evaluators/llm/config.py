"""
Run configuration for TsT-LLM. The defaults are the paper's configuration (App. C.1).
"""

from dataclasses import asdict, dataclass
from typing import Optional

PAPER_MODEL = "Qwen/Qwen2-7B-Instruct"
# The revision the Hub's main pointed to for all of the paper's TsT-LLM runs (since 2024-08-21).
PAPER_MODEL_REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"


@dataclass
class LLMRunConfig:
    """Inference and LoRA-training settings for one TsT-LLM run (defaults: App. C.1)."""

    model_name: str = PAPER_MODEL
    # None: the paper's revision for the paper's model, else the Hub's main ("main" also works).
    model_revision: Optional[str] = None
    max_seq_length: int = 1024

    # Inference (vLLM). Prompts are raw text (no chat template), as in the paper.
    eval_batch_size: int = 4
    temperature: float = 0.0
    max_tokens: int = 10  # generated NUM answers; MC is scored from one token's log-probabilities
    apply_chat_template: bool = False
    use_logprob_scoring: bool = True  # MC score = probability of the gold option
    logprob_top_k: int = 100
    gpu_memory_utilization: float = 0.8

    # LoRA training (LLaMA-Factory). Training formats examples with the "gemma" template.
    learning_rate: float = 2e-4
    train_batch_size: int = 4
    num_epochs: int = 1
    lora_rank: int = 8
    lora_alpha: Optional[int] = 16  # None: 2 * lora_rank
    lora_dropout: float = 0.1
    template: str = "gemma"

    def __post_init__(self):
        if self.model_revision is None and self.model_name == PAPER_MODEL:
            self.model_revision = PAPER_MODEL_REVISION

    def to_dict(self) -> dict:
        return asdict(self)

    def to_predictor_config(self):
        """vLLM predictor settings."""
        from .predictors.vllm import VLLMPredictorConfig

        # vLLM serves LoRA ranks up to max_lora_rank: the next power of two >= the trained rank, at least 16.
        max_lora_rank = max(16, 1 << (self.lora_rank - 1).bit_length()) if self.lora_rank > 0 else 16
        return VLLMPredictorConfig(
            model_name=self.model_name,
            model_revision=self.model_revision,
            max_seq_length=self.max_seq_length,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            batch_size=self.eval_batch_size,
            apply_chat_template=self.apply_chat_template,
            use_logprob_scoring=self.use_logprob_scoring,
            logprob_top_k=self.logprob_top_k,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_lora_rank=max_lora_rank,
        )

    def to_trainer_config(self, seed: int = 42):
        """LLaMA-Factory LoRA settings. ``seed`` is the training seed: random_state + repeat index, the
        same for every fold of a repeat (42 for the paper's single repeat, as in its runs)."""
        from .trainers.llamafactory import LlamaFactoryConfig

        return LlamaFactoryConfig(
            model_name=self.model_name,
            model_revision=self.model_revision,
            template=self.template,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            batch_size=self.train_batch_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            max_seq_length=self.max_seq_length,
            seed=seed,
        )
