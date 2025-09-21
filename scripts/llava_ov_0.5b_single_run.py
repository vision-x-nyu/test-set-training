from TsT.experiments import run_single_llm_experiment
from TsT.evaluators.llm.config import LLMRunConfig


def main():
    """Run single LLM evaluation with fixed hyperparameters."""

    # Configure hyperparameters
    llm_config = LLMRunConfig(
        model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        learning_rate=6e-4,
        train_batch_size=16,
        eval_batch_size=16,
        lora_rank=16,
        lora_alpha=32,
        num_epochs=6,
        max_seq_length=2048,
        temperature=0.0,
        max_tokens=128,
        template="llava_next",
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
