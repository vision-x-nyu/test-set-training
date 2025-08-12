#!/usr/bin/env python3
"""
Example usage of TsT LLM system for bias detection.

This script demonstrates how to use the LLM infrastructure with the
unified evaluation framework for bias detection in multimodal benchmarks.
"""

import os
import pandas as pd
import tempfile

# Import the TsT LLM components
from TsT.evaluators.llm import (
    create_vllm_predictor,
    create_llamafactory_trainer,
    create_trainable_predictor,
    create_auto_predictor,
    get_gpu_count,
)
from TsT.evaluators import LLMEvaluator


def create_sample_data():
    """Create sample data for demonstration"""
    data = {
        "id": range(100),
        "question": [f"What is the answer to question {i}?" for i in range(100)],
        "gt_idx": ["A" if i % 2 == 0 else "B" for i in range(100)],
        "options": [["A: Option A", "B: Option B"] for _ in range(100)],
    }
    return pd.DataFrame(data)


class SampleBiasModel:
    """Sample bias model for demonstration"""

    name = "sample_llm_model"
    format = "mc"  # multiple choice
    target_col_override = None

    def select_rows(self, df):
        return df

    @property
    def task(self):
        return "clf"

    @property
    def metric(self):
        return "acc"


def example_single_gpu_usage():
    """Example: Using single-GPU LLM predictor"""
    print("=== Single-GPU LLM Example ===")

    # Create single-GPU predictor
    predictor = create_vllm_predictor(
        model_name="google/gemma-2-2b-it",
        max_seq_length=256,
        temperature=0.0,  # Deterministic for bias detection
        max_tokens=5,
    )

    # Create trainer
    trainer = create_llamafactory_trainer(
        model_name="google/gemma-2-2b-it", learning_rate=2e-4, num_epochs=1, batch_size=4
    )

    # Create trainable predictor
    trainable = create_trainable_predictor(predictor, trainer)

    print("✅ Created single-GPU trainable predictor")
    print(f"   GPU count: {get_gpu_count()}")
    print(f"   Predictor loaded: {trainable.is_loaded}")

    return trainable


def example_multi_gpu_usage():
    """Example: Using multi-GPU Ray predictor"""
    print("\n=== Multi-GPU Ray Example ===")

    gpu_count = get_gpu_count()
    if gpu_count <= 1:
        print("⚠️  Multi-GPU example skipped (only 1 GPU available)")
        return None

    # Use the auto predictor to choose best option
    predictor = create_auto_predictor(model_name="google/gemma-2-2b-it", prefer_multi_gpu=True)

    trainer = create_llamafactory_trainer(model_name="google/gemma-2-2b-it", num_epochs=1)

    trainable = create_trainable_predictor(predictor, trainer)

    print("✅ Created multi-GPU trainable predictor")
    print(f"   GPU count: {gpu_count}")
    print(f"   Predictor type: {type(predictor).__name__}")

    return trainable


def example_evaluation_integration():
    """Example: Integration with evaluation framework"""
    print("\n=== Evaluation Framework Integration ===")

    # Create sample model and data
    model = SampleBiasModel()  # noqa: F841
    df = create_sample_data()  # noqa: F841

    # Use LLM fold evaluator for LLM evaluation

    llm_evaluator = LLMEvaluator(
        model=model,
        df=df,
        target_col="gt_idx",
        llm_config=None,  # use default config
    )

    print("✅ Created LLM fold evaluator")
    print(f"   Type: {type(llm_evaluator).__name__}")

    # Option 2: Use production LLM evaluator (when ready for full GPU training)
    # trainable = example_single_gpu_usage()
    # from TsT.evaluators import LLMEvaluator
    # llm_evaluator = LLMEvaluator(trainable)

    # Example of running cross-validation (commented out to avoid actual training)
    # print("\n🔄 Running cross-validation...")
    # mean_score, std_score, count = run_cross_validation(
    #     model=model,
    #     evaluator=legacy_evaluator,
    #     df=df,
    #     target_col="gt_idx",
    #     n_splits=2,
    #     random_state=42,
    #     verbose=True
    # )
    # print(f"📊 Results: {mean_score:.3f} ± {std_score:.3f} (n={count})")


def example_data_conversion():
    """Example: Data conversion utilities"""
    print("\n=== Data Conversion Example ===")

    from TsT.evaluators.llm.data.conversion import convert_to_blind_training_format, convert_to_blind_test_instances

    # Create sample benchmark data
    df = create_sample_data()

    # Convert to LLM training format
    training_data = convert_to_blind_training_format(
        df=df[:5],  # Use first 5 rows
        target_col="gt_idx",
        format_type="mc",
    )

    # Convert to test instances
    test_instances = convert_to_blind_test_instances(
        df=df[5:10],  # Use next 5 rows
        target_col="gt_idx",
        format_type="mc",
    )

    print("✅ Converted benchmark data")
    print(f"   Training examples: {len(training_data)}")
    print(f"   Test instances: {len(test_instances)}")
    print(f"   Sample training: {training_data[0].instruction[:50]}...")
    print(f"   Sample response: {training_data[0].response}")


def example_io_utilities():
    """Example: I/O utilities for type-safe JSONL operations"""
    print("\n=== I/O Utilities Example ===")

    from TsT.evaluators.llm.utils.io import write_jsonl, read_jsonl
    from TsT.evaluators.llm.data.models import TrainingDatum

    # Create sample training data
    training_data = [
        TrainingDatum(instruction="What is 2+2?", response="4", metadata={"example": True}),
        TrainingDatum(instruction="What is the capital of France?", response="Paris"),
    ]

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = f.name

    write_jsonl(training_data, temp_path)

    # Read back
    loaded_data = read_jsonl(temp_path, TrainingDatum)

    print("✅ Type-safe JSONL I/O")
    print(f"   Written: {len(training_data)} items")
    print(f"   Loaded: {len(loaded_data)} items")
    print(f"   Types match: {type(loaded_data[0]).__name__}")

    # Cleanup
    os.unlink(temp_path)


def main():
    """Run all examples"""
    print("🚀 TsT LLM System Examples")
    print("=" * 50)

    try:
        # Basic component examples
        example_single_gpu_usage()
        example_multi_gpu_usage()

        # Data processing examples
        example_data_conversion()
        example_io_utilities()

        # Integration examples
        example_evaluation_integration()

        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Run with actual GPU resources for full training")
        print("   2. Integrate with your specific benchmark data")
        print("   3. Use Ray for multi-GPU parallel inference")
        print("   4. Experiment with different model configurations")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("   This is expected without GPU resources and model downloads")


if __name__ == "__main__":
    main()
