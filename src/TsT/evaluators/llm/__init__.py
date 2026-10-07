"""
TsT-LLM: k-fold LoRA fine-tuning of a text-only LLM on a benchmark's own questions.

Needs the ``llm`` extra (``uv sync --frozen --extra llm``): vLLM for inference and
LLaMA-Factory for LoRA training, on one NVIDIA GPU. Importing this package does not import
either; they load when an evaluator is created.
"""

import importlib.util
from typing import Optional

from .config import LLMRunConfig

INSTALL_HINT = "TsT-LLM needs the llm extra (uv sync --frozen --extra llm) on Linux with an NVIDIA GPU"


def missing_requirements() -> Optional[str]:
    """None if the TsT-LLM packages are installed, else a one-line explanation (checked before any
    download). It does not touch CUDA: initializing CUDA in this process would break vLLM's engine
    process."""
    missing = [m for m in ("torch", "vllm", "llamafactory") if importlib.util.find_spec(m) is None]
    if missing:
        return f"{INSTALL_HINT}; not installed: {', '.join(missing)}"
    return None


def __getattr__(name):
    if name == "LLMEvaluator":
        from .evaluator import LLMEvaluator

        return LLMEvaluator
    raise AttributeError(name)


__all__ = ["LLMEvaluator", "LLMRunConfig", "missing_requirements"]
