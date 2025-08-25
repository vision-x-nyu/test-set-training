#!/usr/bin/env python3
"""
Multi-benchmark LLM sweep script.

Demonstrates how to run hyperparameter sweeps across multiple benchmarks
using the TsT.experiments module. This allows testing whether certain
hyperparameter settings generalize well across different benchmarks.
"""

from dataclasses import replace

from TsT.experiments import run_llm_multi_benchmark_sweep
from TsT.evaluators.llm.config import LLMRunConfig


def main():
    """Run multi-benchmark LLM hyperparameter sweep."""

    # Base configuration - we'll vary key hyperparameters from this
    base_config = LLMRunConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        learning_rate=6e-4,
        train_batch_size=8,
        eval_batch_size=16,
        lora_rank=8,
        num_epochs=8,
        max_seq_length=1024,
        temperature=0.0,
        max_tokens=10,
        template="qwen",
    )

    # Define a focused set of configurations to test across benchmarks
    configs = []

    # 1. Core configurations around the best settings found in single-benchmark sweeps
    core_configs = [
        # Best performer from single-benchmark (baseline)
        dict(learning_rate=6e-4, train_batch_size=8, lora_rank=8, num_epochs=4),
        dict(learning_rate=6e-4, train_batch_size=8, lora_rank=8, num_epochs=8),
    ]

    for config_dict in core_configs:
        configs.append(replace(base_config, **config_dict))

    # Define benchmarks to test across
    benchmarks = ["cvb", "vsi", "mmmu", "video_mme"]

    print(f"🚀 Starting multi-benchmark sweep with {len(configs)} configurations")
    print(f"   Benchmarks: {', '.join(benchmarks)}")
    print(f"   Total runs: {len(configs)} × {len(benchmarks)} = {len(configs) * len(benchmarks)}")

    sweep_dir = run_llm_multi_benchmark_sweep(
        configs=configs,
        benchmarks=benchmarks,
        experiment_name="qwen2.5_7b_multi_benchmark_test_v2",
        n_splits=2,
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
