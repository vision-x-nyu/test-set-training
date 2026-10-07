"""
vLLM predictor with LoRA adapters (one GPU).

For multiple-choice instances it generates one token with the top log-probabilities and
records the probability of each option letter; for other instances it generates up to
``max_tokens`` tokens. Prompts are raw text unless ``apply_chat_template`` is set.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..data.models import LLMPredictionResult, TestInstance
from ..scoring import gold_letter, option_probs_from_logprobs, option_token_ids
from .base import LLMPredictorInterface, validate_instances

logger = logging.getLogger(__name__)


@dataclass
class VLLMPredictorConfig:
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    model_revision: Optional[str] = None
    max_seq_length: int = 1024
    temperature: float = 0.0
    max_tokens: int = 10
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.8
    enable_lora: bool = True
    max_lora_rank: int = 16  # must be >= the trained LoRA rank
    batch_size: int = 4

    apply_chat_template: bool = False
    system_prompt: str = "You are a helpful assistant that answers questions accurately and concisely."

    use_logprob_scoring: bool = True
    logprob_top_k: int = 100


class VLLMPredictor(LLMPredictorInterface):
    def __init__(self, config: VLLMPredictorConfig, enforce_eager: bool = False):
        from vllm import SamplingParams

        self.config = config
        self.enforce_eager = enforce_eager
        self.tokenizer = None
        self.llm = None
        self.lora_request = None
        self._adapter_path: Optional[str] = None

        self.sampling_params = SamplingParams(
            temperature=config.temperature, max_tokens=config.max_tokens, top_p=config.top_p, stop=None
        )
        self.logprob_sampling_params = (
            SamplingParams(temperature=0.0, max_tokens=1, logprobs=config.logprob_top_k)
            if config.use_logprob_scoring
            else None
        )
        self._load_base_model()

    def __str__(self) -> str:
        return f"VLLMPredictor(config={self.config})"

    @property
    def is_loaded(self) -> bool:
        return self.llm is not None

    @property
    def current_adapter_path(self) -> Optional[str]:
        return self._adapter_path

    def _load_base_model(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, revision=self.config.model_revision)
            llm_kwargs = dict(
                model=self.config.model_name,
                revision=self.config.model_revision,
                enable_lora=self.config.enable_lora,
                max_model_len=self.config.max_seq_length,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
            )
            if self.config.enable_lora:
                llm_kwargs["max_lora_rank"] = self.config.max_lora_rank
            if self.config.use_logprob_scoring:
                llm_kwargs["max_logprobs"] = self.config.logprob_top_k
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            self.llm = None
            raise RuntimeError(f"Failed to load base model {self.config.model_name}: {e}") from e

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            self._load_base_model()

    def load_adapter(self, adapter_path: str) -> None:
        from transformers import AutoTokenizer
        from vllm.lora.request import LoRARequest

        if not self.is_loaded:
            raise RuntimeError("Base model not loaded")
        path = Path(adapter_path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        # The tokenizer saved with the adapter is the one used in training.
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        # The base model is reloaded before every adapter (see reset), so one adapter id suffices.
        self.lora_request = LoRARequest(lora_name=f"tst_adapter_{path.name}", lora_int_id=1, lora_path=adapter_path)
        self._adapter_path = adapter_path

    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        if not instances:
            return []
        validate_instances(instances)

        use_logprobs = self.logprob_sampling_params is not None
        mc_idx = [i for i, inst in enumerate(instances) if use_logprobs and inst.options]
        gen_idx = [i for i in range(len(instances)) if i not in set(mc_idx)]
        results: List[Optional[LLMPredictionResult]] = [None] * len(instances)

        if mc_idx:
            mc_instances = [instances[i] for i in mc_idx]
            outputs = self._generate([inst.instruction for inst in mc_instances], self.logprob_sampling_params)
            for i, inst, output in zip(mc_idx, mc_instances, outputs):
                text = output.outputs[0].text.strip()
                option_probs, confidence = None, None
                if output.outputs[0].logprobs:
                    logprobs = {tid: lp.logprob for tid, lp in output.outputs[0].logprobs[0].items()}
                    option_probs = option_probs_from_logprobs(logprobs, option_token_ids(self.tokenizer, inst.options))
                    if option_probs is not None:
                        confidence = option_probs.get(gold_letter(inst.ground_truth, inst.options))
                if confidence is None:
                    logger.info(
                        f"{inst.instance_id}: gold letter not among the top log-probabilities; parsing the text"
                    )
                results[i] = LLMPredictionResult(
                    instance_id=inst.instance_id,
                    prediction=_first_token(text),
                    confidence=confidence,
                    option_probs=option_probs,
                    raw_output=text,
                )

        if gen_idx:
            gen_instances = [instances[i] for i in gen_idx]
            outputs = self._generate([inst.instruction for inst in gen_instances], self.sampling_params)
            for i, inst, output in zip(gen_idx, gen_instances, outputs):
                text = output.outputs[0].text.strip()
                results[i] = LLMPredictionResult(
                    instance_id=inst.instance_id, prediction=_first_token(text), raw_output=text
                )

        return results  # type: ignore[return-value]

    def _generate(self, instructions: List[str], sampling_params):
        prompts = self._format_prompts(instructions)
        outputs = []
        for start in range(0, len(prompts), self.config.batch_size):
            outputs.extend(
                self.llm.generate(
                    prompts[start : start + self.config.batch_size],
                    sampling_params,
                    use_tqdm=False,
                    lora_request=self.lora_request,
                )
            )
        return outputs

    def _format_prompts(self, instructions: List[str]) -> List[str]:
        if not self.config.apply_chat_template or self.tokenizer.chat_template is None:
            return instructions
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.config.system_prompt}, {"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for instruction in instructions
        ]

    def reset(self) -> None:
        self.lora_request = None
        self._adapter_path = None
        self.llm = None
        self.tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


def _first_token(generated_text: str) -> str:
    """The first whitespace-separated token, without trailing punctuation."""
    tokens = generated_text.split()
    if not tokens:
        return generated_text
    return tokens[0].rstrip(".,!?;:")
