#!/usr/bin/env python3
"""
Multi-benchmark LLM sweep script (v3 - expanded sweep).

Runs a substantially larger hyperparameter sweep across multiple
benchmarks, informed by previous single- and multi-benchmark results.
"""

from dataclasses import replace

from TsT.experiments import run_llm_multi_benchmark_sweep
from TsT.evaluators.llm.config import LLMRunConfig


def main():
    """Run multi-benchmark LLM hyperparameter sweep (v3 - expanded)."""

    # Base configuration - we'll vary key hyperparameters from this
    base_config = LLMRunConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        learning_rate=6e-4,
        train_batch_size=8,
        eval_batch_size=16,
        lora_rank=8,
        num_epochs=12,
        max_seq_length=1024,
        temperature=0.0,
        max_tokens=10,
        template="qwen",
        lora_dropout=0.1,
        lora_alpha=None,
    )

    # Build a large pool of configurations
    configs = []

    # 1) (Optional) Hand-picked variants can be added here if needed
    # Intentionally left empty to keep total runs ~half of the original v3

    # 2) Moderately broad grid around promising regions
    learning_rates = [3e-4, 4e-4, 6e-4, 8e-4, 1e-4]
    train_batch_sizes = [8, 16]
    lora_ranks = [8, 16]
    num_epochs_list = [12, 16, 24]
    for lr in learning_rates:
        for bs in train_batch_sizes:
            for rank in lora_ranks:
                for epochs in num_epochs_list:
                    configs.append(
                        replace(
                            base_config,
                            learning_rate=lr,
                            train_batch_size=bs,
                            lora_rank=rank,
                            num_epochs=epochs,
                        )
                    )

    # 3) Regularization-focused sweep on a compact set of top variants
    top_variants = [
        (8e-4, 8, 8, 12),
        (4e-4, 8, 16, 24),
        (3e-4, 8, 8, 24),
        (6e-4, 16, 8, 16),
    ]
    for lr, bs, rank, epochs in top_variants:
        for dropout in [0.05, 0.1, 0.2]:
            for alpha_mult in [1, 2]:
                configs.append(
                    replace(
                        base_config,
                        learning_rate=lr,
                        train_batch_size=bs,
                        lora_rank=rank,
                        num_epochs=epochs,
                        lora_dropout=dropout,
                        lora_alpha=rank * alpha_mult,
                    )
                )

    # Deduplicate configurations
    deduped = []
    seen = set()
    for c in configs:
        key = (
            c.learning_rate,
            c.train_batch_size,
            c.lora_rank,
            c.num_epochs,
            c.lora_alpha,
            c.lora_dropout,
        )
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    configs = deduped

    # Define benchmarks to test across
    benchmarks = ["cv-bench", "vsi", "mmmu", "video_mme"]

    print(f"🚀 Starting multi-benchmark sweep (v3) with {len(configs)} configurations")
    print(f"   Benchmarks: {', '.join(benchmarks)}")
    print(f"   Total runs: {len(configs)} × {len(benchmarks)} = {len(configs) * len(benchmarks)}")

    sweep_dir = run_llm_multi_benchmark_sweep(
        configs=configs,
        benchmarks=benchmarks,
        experiment_name="qwen2.5_7b_multi_benchmark_sweep_v3",
        n_splits=3,
        random_state=42,
        verbose=True,
        continue_on_failure=True,
        resume=True,  # Enable resume by default
    )

    print(f"\n✅ Multi-benchmark sweep complete! Results in: {sweep_dir}")
    print("\n📊 Summary files generated:")
    print("   • benchmark_summary.csv - Best performance per benchmark (with config)")
    print("   • config_summary.csv - Generalization stats per config")
    print("   • sweep_summary.csv - All raw results")


if __name__ == "__main__":
    main()
