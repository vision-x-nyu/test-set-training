#!/usr/bin/env python3
"""
Lightweight LLM single run script.

Demonstrates how to run a single LLM experiment with fixed hyperparameters
using the TsT.experiments module.
"""

from TsT.experiments import run_single_llm_experiment
from TsT.evaluators.llm.config import LLMRunConfig


def main():
    """Run single LLM evaluation with fixed hyperparameters."""

    # Configure hyperparameters
    llm_config = LLMRunConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        learning_rate=8e-4,
        train_batch_size=8,
        eval_batch_size=16,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0,
        num_epochs=12,
        max_seq_length=1024,
        temperature=0.0,
        max_tokens=10,
        template="qwen",
    )

    print("🚀 Starting single LLM experiment")

    # Run experiment
    results_dir = run_single_llm_experiment(
        llm_config=llm_config,
        benchmark="video_mme",
        n_splits=3,  # Use fewer splits for faster testing
        random_state=42,
        verbose=True,
    )

    print(f"\n✅ Experiment complete! Results in: {results_dir}")


if __name__ == "__main__":
    main()
