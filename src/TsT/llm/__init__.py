"""
TsT LLM package for production-ready LLM bias detection.

This package provides a complete LLM infrastructure for bias detection following
DataEnvGym patterns, including:

- Type-safe data models with Pydantic
- Single and multi-GPU vLLM predictors
- LlamaFactory-based LoRA training
- Composed trainable predictors
- Integration with the unified evaluation framework

Example usage:

    # Create single-GPU predictor
    from TsT.llm import create_vllm_predictor, create_llamafactory_trainer, create_trainable_predictor

    predictor = create_vllm_predictor("google/gemma-2-2b-it")
    trainer = create_llamafactory_trainer("google/gemma-2-2b-it")
    trainable = create_trainable_predictor(predictor, trainer)

    # Use with evaluation framework
    from TsT.evaluators import LLMEvaluator
    # Note: LLMEvaluator requires model, df, target_col, and llm_config parameters
"""

# Data models
from .data.models import (
    TrainingDatum,
    TestInstance,
    LLMPredictionResult,
    LoRAAdapterInfo,
    LLMEvaluationMetrics,
    LLMConfig,
)

# Data conversion utilities
from .data.conversion import (
    convert_to_tst_training_format,
    convert_to_test_instances,
    convert_benchmark_to_llm_format,
    format_for_chat_template,
)

# Base interfaces
from .predictors.base import LLMPredictorInterface, BaseLLMPredictor
from .trainers.base import LLMTrainerInterface, BaseLLMTrainer, ProgressCallback

# vLLM predictors
from .predictors.vllm import VLLMPredictor, VLLMPredictorConfig
from .predictors.ray_vllm import RayVLLMPredictor, RayVLLMPredictorConfig

# LlamaFactory trainer
from .trainers.llamafactory import (
    LlamaFactoryTrainer,
    LlamaFactoryConfig,
    TrainingProgressMonitor,
    create_llamafactory_trainer,
)

# Composed trainable predictor
from .trainable.predictor import (
    TrainableLLMPredictor,
    TrainableLLMPredictorConfig,
    create_trainable_predictor,
)

# I/O utilities
from .utils.io import (
    PydanticJSONLinesWriter,
    PydanticJSONLinesReader,
    write_jsonl,
    read_jsonl,
)


def create_vllm_predictor(
    model_name: str = "google/gemma-2-2b-it",
    max_seq_length: int = 512,
    temperature: float = 0.0,
    max_tokens: int = 10,
    **kwargs,
) -> VLLMPredictor:
    """
    Create a single-GPU vLLM predictor with common settings.

    Args:
        model_name: Model name to load
        max_seq_length: Maximum sequence length
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration parameters

    Returns:
        Configured VLLMPredictor instance
    """
    config = VLLMPredictorConfig(
        model_name=model_name, max_seq_length=max_seq_length, temperature=temperature, max_tokens=max_tokens, **kwargs
    )
    return VLLMPredictor(config)


def create_ray_vllm_predictor(
    model_name: str = "google/gemma-2-2b-it",
    num_workers: int = 4,
    max_seq_length: int = 512,
    temperature: float = 0.0,
    max_tokens: int = 10,
    **kwargs,
) -> RayVLLMPredictor:
    """
    Create a multi-GPU Ray vLLM predictor with common settings.

    Args:
        model_name: Model name to load
        num_workers: Number of Ray workers (GPUs)
        max_seq_length: Maximum sequence length
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration parameters

    Returns:
        Configured RayVLLMPredictor instance
    """
    base_config = VLLMPredictorConfig(
        model_name=model_name, max_seq_length=max_seq_length, temperature=temperature, max_tokens=max_tokens, **kwargs
    )

    ray_config = RayVLLMPredictorConfig(base_config=base_config, num_workers=num_workers)

    return RayVLLMPredictor(ray_config)


def get_gpu_count() -> int:
    """
    Get the number of available GPUs.

    Returns:
        Number of available GPUs
    """
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def create_auto_predictor(
    model_name: str = "google/gemma-2-2b-it", prefer_multi_gpu: bool = True, **kwargs
) -> LLMPredictorInterface:
    """
    Automatically create the best predictor based on available hardware.

    Args:
        model_name: Model name to load
        prefer_multi_gpu: Whether to prefer multi-GPU if available
        **kwargs: Additional configuration parameters

    Returns:
        Best available predictor instance
    """
    gpu_count = get_gpu_count()

    if gpu_count > 1 and prefer_multi_gpu:
        return create_ray_vllm_predictor(
            model_name=model_name,
            num_workers=min(gpu_count, 4),  # Cap at 4 workers
            **kwargs,
        )
    else:
        return create_vllm_predictor(model_name=model_name, **kwargs)


# Export all important classes and functions
__all__ = [
    # Data models
    "TrainingDatum",
    "TestInstance",
    "LLMPredictionResult",
    "LoRAAdapterInfo",
    "LLMEvaluationMetrics",
    "LLMConfig",
    # Conversion utilities
    "convert_to_tst_training_format",
    "convert_to_test_instances",
    "convert_benchmark_to_llm_format",
    "format_for_chat_template",
    # Base interfaces
    "LLMPredictorInterface",
    "BaseLLMPredictor",
    "LLMTrainerInterface",
    "BaseLLMTrainer",
    "ProgressCallback",
    # Predictors
    "VLLMPredictor",
    "VLLMPredictorConfig",
    "RayVLLMPredictor",
    "RayVLLMPredictorConfig",
    # Trainers
    "LlamaFactoryTrainer",
    "LlamaFactoryConfig",
    "TrainingProgressMonitor",
    "create_llamafactory_trainer",
    # Trainable predictor
    "TrainableLLMPredictor",
    "TrainableLLMPredictorConfig",
    "create_trainable_predictor",
    # I/O utilities
    "PydanticJSONLinesWriter",
    "PydanticJSONLinesReader",
    "write_jsonl",
    "read_jsonl",
    # Factory functions
    "create_vllm_predictor",
    "create_ray_vllm_predictor",
    "create_auto_predictor",
    "get_gpu_count",
]
