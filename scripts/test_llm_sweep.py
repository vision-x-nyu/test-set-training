#!/usr/bin/env python3
"""
Lightweight LLM hyperparameter sweep script.

Demonstrates how to run hyperparameter sweeps with different configurations
using the TsT.experiments module.
"""

from TsT.experiments import run_llm_sweep
from TsT.evaluators.llm.config import LLMRunConfig


def main():
    """Run LLM hyperparameter sweep with multiple configurations."""

    # Define hyperparameter grid
    configs = [
        # Small, fast configurations for testing
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=2e-4,
            train_batch_size=16,  # Smaller batch size to avoid OOM
            lora_rank=8,
            lora_alpha=16,
            num_epochs=5,
            max_seq_length=1024,
            eval_batch_size=4,
        ),
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=5e-4,
            train_batch_size=16,
            lora_rank=16,
            lora_alpha=32,
            num_epochs=5,
            max_seq_length=1024,
            eval_batch_size=4,
        ),
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=1e-3,
            train_batch_size=8,  # Even smaller for higher LR
            lora_rank=32,
            lora_alpha=64,
            num_epochs=5,
            max_seq_length=1024,
            eval_batch_size=4,
        ),
    ]

    print(f"🚀 Starting hyperparameter sweep with {len(configs)} configurations")

    # Run sweep
    sweep_dir = run_llm_sweep(
        configs=configs,
        benchmark="mmmu",
        n_splits=2,  # Use fewer splits for faster testing
        random_state=42,
        verbose=True,
        continue_on_failure=True,  # Don't stop if one config fails
    )

    print(f"\n✅ Sweep complete! Results in: {sweep_dir}")


if __name__ == "__main__":
    main()
