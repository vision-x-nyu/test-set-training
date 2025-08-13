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

    # Make bs=8 the default; we’ll override per-config when needed.
    base_config = LLMRunConfig(
        model_name="Qwen/Qwen2-7B-Instruct",
        learning_rate=6e-4,
        train_batch_size=8,  # << smaller default to increase updates/epoch
        eval_batch_size=16,
        lora_rank=16,
        num_epochs=8,  # << paired with bs=8; overridden below as needed
        max_seq_length=1024,
        temperature=0.0,
        max_tokens=10,
        template="qwen",
    )

    configs = []

    # --- 1) Matched-total-steps fairness set (≈500 updates each) at r=8 ---
    # steps/epoch ≈ ceil(500/bs); keep ~512 updates for fair comparison.
    matched = [
        dict(train_batch_size=32, num_epochs=32, lora_rank=8, learning_rate=6e-4),  # +8.1%
        dict(train_batch_size=16, num_epochs=16, lora_rank=8, learning_rate=6e-4),  # +6.9%
        dict(train_batch_size=8, num_epochs=8, lora_rank=8, learning_rate=6e-4),  # +10.6%
    ]
    for kw in matched:
        configs.append(replace(base_config, **kw))

    # --- 2) Core sweep anchored at bs=8 (good for small data + bias surfacing) ---
    core_bs8 = [
        # (rank, lr, epochs)
        (4, 6e-4, 24),  # +2.8%
        (8, 6e-4, 16),  # +6.6%
        (8, 1e-3, 16),  # +2.6%
        (16, 4e-4, 16),  # +5.4%
        (16, 2e-4, 24),  # +2.5%
        (32, 2e-4, 16),
        (32, 1e-4, 24),
    ]
    for r, lr, ep in core_bs8:
        configs.append(
            replace(
                base_config,
                lora_rank=r,
                learning_rate=lr,
                num_epochs=ep,
                train_batch_size=8,
            )
        )

    # --- 3) Fine sweep around the sweet spot (r=8, bs=8) ---
    for lr in [3e-4, 4e-4, 6e-4, 8e-4]:
        for ep in [12, 24]:
            configs.append(
                replace(
                    base_config,
                    lora_rank=8,
                    learning_rate=lr,
                    num_epochs=ep,
                    train_batch_size=8,
                )
            )

    # --- 4) Context length ablation (can expose shortcut/bias behavior) ---
    configs.append(
        replace(
            base_config,
            lora_rank=8,
            learning_rate=6e-4,
            num_epochs=24,
            max_seq_length=512,
            train_batch_size=8,
        )
    )

    # --- 5) Probe runs: push memorization a bit (bs=4 variants) ---
    configs.append(
        replace(
            base_config,
            lora_rank=16,
            learning_rate=4e-4,
            num_epochs=16,
            train_batch_size=4,
        )
    )
    configs.append(
        replace(
            base_config,
            lora_rank=16,
            learning_rate=4e-4,
            num_epochs=24,
            train_batch_size=8,
        )
    )
    configs.append(
        replace(
            base_config,
            lora_rank=32,
            learning_rate=3e-4,
            num_epochs=24,
            train_batch_size=4,
        )
    )

    print(f"🚀 Starting hyperparameter sweep with {len(configs)} configurations")

    sweep_dir = run_llm_sweep(
        configs=configs,
        benchmark="mmmu",
        experiment_name="qwen2_7b_mmmu_sweep",
        n_splits=3,
        random_state=42,
        verbose=True,
        continue_on_failure=True,
        resume=True,  # Enable resume by default
    )

    print(f"\n✅ Sweep complete! Results in: {sweep_dir}")


if __name__ == "__main__":
    main()
