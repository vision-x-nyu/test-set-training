#!/usr/bin/env python3
"""
Lightweight LLM hyperparameter sweep script.

Demonstrates how to run hyperparameter sweeps with different configurations
using the TsT.experiments module.
"""

from dataclasses import replace

from TsT.experiments import run_llm_sweep
from TsT.evaluators.llm.config import LLMRunConfig


def main():
    """Run LLM hyperparameter sweep with multiple configurations."""

    base_config = LLMRunConfig(
        model_name="Qwen/Qwen2-7B-Instruct",
        learning_rate=6e-4,
        train_batch_size=32,
        eval_batch_size=16,
        lora_rank=16,
        num_epochs=4,
        max_seq_length=1024,
        temperature=0.0,
        max_tokens=10,
        template="qwen",
    )

    # ---- CURATED SWEEP (19 runs) ----
    configs = []

    # Coarse tradeoff: (rank, lr, epochs)
    coarse = [
        (4, 6e-4, 32),
        (4, 1e-3, 16),
        (8, 6e-4, 16),
        (8, 1e-3, 16),
        (16, 4e-4, 16),
        (16, 2e-4, 8),
        (32, 2e-4, 8),
        (32, 1e-4, 16),
        (8, 4e-4, 32),
    ]
    for r, lr, ep in coarse:
        configs.append(
            replace(
                base_config,
                lora_rank=r,
                learning_rate=lr,
                num_epochs=ep,
            )
        )

    # Fine sweep around the most likely sweet spot (r=8)
    for lr in [3e-4, 4e-4, 6e-4, 8e-4]:
        for ep in [12, 24]:
            configs.append(
                replace(
                    base_config,
                    lora_rank=8,
                    learning_rate=lr,
                    num_epochs=ep,
                )
            )

    # Context length ablation (can expose shortcut/bias behavior)
    configs.append(
        replace(
            base_config,
            lora_rank=8,
            learning_rate=6e-4,
            num_epochs=24,
            max_seq_length=512,
        )
    )

    # High-capacity but conservative LR to intentionally overfit a bit
    configs.append(
        replace(
            base_config,
            lora_rank=32,
            learning_rate=3e-4,
            num_epochs=24,
        )
    )

    print(f"🚀 Starting hyperparameter sweep with {len(configs)} configurations")

    # Run sweep
    sweep_dir = run_llm_sweep(
        configs=configs,
        benchmark="mmmu",
        # n_splits=2,  # Use fewer splits for faster testing
        n_splits=3,  # Use fewer splits for faster testing
        random_state=42,
        verbose=True,
        continue_on_failure=True,  # Don't stop if one config fails
    )

    print(f"\n✅ Sweep complete! Results in: {sweep_dir}")


if __name__ == "__main__":
    main()
