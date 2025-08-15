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

    MAX_SEQ_LENGTH = 2048

    # NOTE: vllm max lora rank is 16?

    # Define hyperparameter grid - OPTIMIZED CONFIGURATIONS
    configs = [
        # HIGH PRIORITY: Fine-tune around the best performer (30.8%)
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=4e-4,  # Slightly lower than best
            train_batch_size=16,
            lora_rank=16,
            lora_alpha=32,  # Keep best ratio
            lora_dropout=0.05,  # Reduce overfitting
            num_epochs=6,  # Try more epochs
            max_seq_length=MAX_SEQ_LENGTH,
            eval_batch_size=4,
        ),
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=6e-4,  # Slightly higher than best
            train_batch_size=20,  # Larger batch
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            num_epochs=4,  # Fewer epochs for higher LR
            max_seq_length=MAX_SEQ_LENGTH,
            eval_batch_size=4,
        ),
        # MEDIUM PRIORITY: Better LoRA configurations
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=5e-4,  # Best LR
            train_batch_size=16,
            lora_rank=24,  # Between 16 and 32
            lora_alpha=48,  # Maintain 2:1 ratio
            lora_dropout=0.08,
            num_epochs=5,
            max_seq_length=MAX_SEQ_LENGTH,
            eval_batch_size=4,
        ),
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=3e-4,  # Lower but stable
            train_batch_size=24,  # Larger batch for stability
            lora_rank=32,  # Higher rank with lower LR
            lora_alpha=64,  # Maintain 2:1 ratio
            lora_dropout=0.1,
            num_epochs=6,  # More epochs for lower LR
            max_seq_length=MAX_SEQ_LENGTH,
            eval_batch_size=4,
        ),
        # EXPERIMENTAL: Advanced configurations
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=5e-4,
            train_batch_size=12,  # Smaller batch for stability
            lora_rank=20,  # Intermediate rank
            lora_alpha=32,  # Lower ratio (1.6:1) for regularization
            lora_dropout=0.12,  # Higher dropout
            num_epochs=7,  # More training
            max_seq_length=MAX_SEQ_LENGTH,
            eval_batch_size=4,
        ),
        LLMRunConfig(
            model_name="google/gemma-2-2b-it",
            learning_rate=7e-4,  # Aggressive but not 1e-3
            train_batch_size=8,  # Small batch for stability
            lora_rank=12,  # Lower rank for regularization
            lora_alpha=24,  # 2:1 ratio
            lora_dropout=0.15,  # High dropout for generalization
            num_epochs=3,  # Few epochs for high LR
            max_seq_length=MAX_SEQ_LENGTH,
            eval_batch_size=4,
        ),
    ]

    print(f"🚀 Starting hyperparameter sweep with {len(configs)} configurations")

    # Run sweep
    sweep_dir = run_llm_sweep(
        configs=configs,
        benchmark="mmmu",
        experiment_name="gemma2_2b_mmmu_sweep",
        n_splits=2,  # Use fewer splits for faster testing
        random_state=42,
        verbose=True,
        continue_on_failure=True,  # Don't stop if one config fails
        resume=True,
    )

    print(f"\n✅ Sweep complete! Results in: {sweep_dir}")


if __name__ == "__main__":
    main()
