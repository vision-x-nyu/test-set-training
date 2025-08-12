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
        model_name="google/gemma-2-2b-it",
        learning_rate=2e-4,
        train_batch_size=16,  # Reduced from 64 to avoid OOM
        lora_rank=16,
        lora_alpha=32,
        num_epochs=1,
        max_seq_length=1024,
        eval_batch_size=4,
        temperature=0.0,
        max_tokens=10,
    )

    print("🚀 Starting single LLM experiment")

    # Run experiment
    results_dir = run_single_llm_experiment(
        llm_config=llm_config,
        benchmark="mmmu",
        n_splits=2,  # Use fewer splits for faster testing
        random_state=42,
        verbose=True,
    )

    print(f"\n✅ Experiment complete! Results in: {results_dir}")


if __name__ == "__main__":
    main()
