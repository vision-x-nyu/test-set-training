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
        train_batch_size=32,
        lora_rank=16,
        num_epochs=2,
        learning_rate=2e-4,
        lora_alpha=32,  # 2x lora_rank as typical
        max_seq_length=1024,
        eval_batch_size=4,
        temperature=0.0,
        max_tokens=10,
    )

    # Run experiment
    results_dir = run_single_llm_experiment(
        llm_config=llm_config,
        benchmark="mmmu",
        n_splits=3,  # Use fewer splits for faster testing
        random_state=42,
        verbose=True,
    )

    print(f"\n✅ Experiment complete! Results in: {results_dir}")


if __name__ == "__main__":
    main()
