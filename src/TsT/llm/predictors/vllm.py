"""
Single-GPU vLLM predictor with LoRA support.

This module implements a vLLM-based predictor for LLM inference with LoRA adapter support,
following DataEnvGym patterns for efficiency and robustness.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .base import BaseLLMPredictor
from ..data.models import TestInstance, LLMPredictionResult


@dataclass
class VLLMPredictorConfig:
    """Configuration for vLLM predictor"""

    model_name: str = "google/gemma-2-2b-it"
    max_seq_length: int = 512
    temperature: float = 0.0
    max_tokens: int = 10
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.8
    enable_lora: bool = True
    batch_size: int = 16

    # Chat template settings
    apply_chat_template: bool = True
    system_prompt: str = "You are a helpful assistant that answers questions accurately and concisely."


class VLLMPredictor(BaseLLMPredictor):
    """Single-GPU vLLM predictor with LoRA support"""

    def __init__(self, config: VLLMPredictorConfig):
        super().__init__()
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm: Optional[LLM] = None
        self.lora_request: Optional[LoRARequest] = None

        # Create sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=None,  # Let the model decide when to stop
        )

        # Load base model
        self._load_base_model()

    def __str__(self) -> str:
        return f"VLLMPredictor(config={self.config})"

    def _load_base_model(self) -> None:
        """Load base model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Load vLLM model
            self.llm = LLM(
                model=self.config.model_name,
                enable_lora=self.config.enable_lora,
                max_model_len=self.config.max_seq_length,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,  # May be needed for some models
            )

            self._set_loaded(True)

        except Exception as e:
            self._set_loaded(False)
            raise RuntimeError(f"Failed to load base model {self.config.model_name}: {e}")

    def load_adapter(self, adapter_path: str) -> None:
        """Load LoRA adapter for fine-tuned inference"""
        if not self.is_loaded:
            raise RuntimeError("Base model not loaded")

        try:
            adapter_path_obj = Path(adapter_path)
            if not adapter_path_obj.exists():
                raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

            # Update tokenizer with any new tokens from fine-tuning
            # This ensures we have the same tokenizer used during training
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

            # Create LoRA request for vLLM
            self.lora_request = LoRARequest(
                lora_name=f"tst_adapter_{adapter_path_obj.name}",
                lora_int_id=1,  # Use ID 1 for our adapter
                lora_path=adapter_path,
            )

            self._set_adapter_path(adapter_path)

        except Exception as e:
            self.lora_request = None
            self._set_adapter_path(None)
            raise RuntimeError(f"Failed to load adapter from {adapter_path}: {e}")

    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """Generate predictions using vLLM"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if not instances:
            return []

        # Validate instances
        self._validate_instances(instances)

        try:
            # Format prompts
            prompts = self._format_prompts([inst.instruction for inst in instances])

            # Generate responses in batches
            all_outputs = []
            for i in range(0, len(prompts), self.config.batch_size):
                batch_prompts = prompts[i : i + self.config.batch_size]

                # Generate using vLLM
                outputs = self.llm.generate(
                    batch_prompts,
                    self.sampling_params,
                    use_tqdm=False,
                    lora_request=self.lora_request,
                )
                all_outputs.extend(outputs)

            # Process outputs into results
            results = []
            for instance, output in zip(instances, all_outputs):
                # Extract the generated text
                generated_text = output.outputs[0].text.strip()

                # For bias detection, we usually want just the first token/word
                # as the prediction (e.g., "A", "B", "C", "D" for MC questions)
                prediction = self._extract_prediction(generated_text)

                result = self._create_prediction_result(
                    instance=instance, prediction=prediction, raw_output=generated_text
                )
                results.append(result)

            return results

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def _format_prompts(self, instructions: List[str]) -> List[str]:
        """Format instructions using chat template if available"""
        if not self.config.apply_chat_template or self.tokenizer.chat_template is None:
            return instructions

        formatted_prompts = []
        for instruction in instructions:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": instruction},
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted_prompt)

        return formatted_prompts

    def _extract_prediction(self, generated_text: str) -> str:
        """
        Extract the actual prediction from generated text.

        For bias detection tasks, we typically want just the first meaningful token.
        """
        if not generated_text:
            return ""

        # Split by whitespace and get the first non-empty token
        tokens = generated_text.split()
        if not tokens:
            return generated_text

        first_token = tokens[0]

        # Remove common punctuation that might be attached
        first_token = first_token.rstrip(".,!?;:")

        return first_token

    def reset(self) -> None:
        """Reset model state and free GPU memory"""
        # Clear LoRA adapter
        self.lora_request = None
        self._set_adapter_path(None)

        # Delete model and tokenizer
        if self.llm is not None:
            del self.llm
            self.llm = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._set_loaded(False)

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.reset()
        except Exception:
            # Ignore cleanup errors during destruction
            pass
