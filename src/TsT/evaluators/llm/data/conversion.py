"""
Dataset format conversion utilities for TsT LLM training.

This module provides functions to convert between different data formats
used in the TsT framework and LLM training pipelines.
"""

import pandas as pd
from typing import List, Literal, Dict, Optional
from .models import TrainingDatum, TestInstance


def convert_to_blind_training_format(
    df: pd.DataFrame,
    target_col: str,
    format_type: Literal["mc", "num"],
    instruction_template: str = "Answer the following question: {question}",
    response_template: str = "The answer is {answer}.",
) -> List[TrainingDatum]:
    """
    Convert DataFrame to TsT training format for LLM fine-tuning.

    Args:
        df: DataFrame with question data
        target_col: Column containing the correct answers
        format_type: Either "mc" for multiple choice or "num" for numerical
        instruction_template: Template for formatting instructions
        response_template: Template for formatting responses

    Returns:
        List of TrainingDatum objects
    """
    training_data = []

    for idx, row in df.iterrows():
        # Extract question text - try different common column names
        question_text = None
        for col in ["question", "text", "instruction", "query"]:
            if col in row and pd.notna(row[col]):
                question_text = str(row[col])
                break

        if question_text is None:
            # Fall back to using all non-target columns as context
            context_cols = [col for col in df.columns if col != target_col]
            question_text = " ".join([f"{col}: {row[col]}" for col in context_cols if pd.notna(row[col])])

        # Format instruction
        instruction = instruction_template.format(question=question_text)

        # Format response based on type
        answer = str(row[target_col])
        if format_type == "mc":
            response = response_template.format(answer=answer)
        else:  # numerical
            response = response_template.format(answer=answer)

        training_data.append(
            TrainingDatum(
                instruction=instruction,
                response=response,
                metadata={
                    "row_id": idx,
                    "format_type": format_type,
                    "target_col": target_col,
                    "original_answer": answer,
                },
            )
        )

    return training_data


def convert_to_test_instances(
    df: pd.DataFrame,
    target_col: str,
    instruction_template: str = "Answer the following question: {question}",
    id_prefix: str = "test",
) -> List[TestInstance]:
    """
    Convert DataFrame to test instances for LLM inference.

    Args:
        df: DataFrame with test data
        target_col: Column containing ground truth answers
        instruction_template: Template for formatting instructions
        id_prefix: Prefix for instance IDs

    Returns:
        List of TestInstance objects
    """
    test_instances = []

    for idx, row in df.iterrows():
        # Extract question text
        question_text = None
        for col in ["question", "text", "instruction", "query"]:
            if col in row and pd.notna(row[col]):
                question_text = str(row[col])
                break

        if question_text is None:
            context_cols = [col for col in df.columns if col != target_col]
            question_text = " ".join([f"{col}: {row[col]}" for col in context_cols if pd.notna(row[col])])

        instruction = instruction_template.format(question=question_text)
        instance_id = f"{id_prefix}_{idx}"
        ground_truth = str(row[target_col])

        test_instances.append(TestInstance(instruction=instruction, instance_id=instance_id, ground_truth=ground_truth))

    return test_instances


def format_for_chat_template(
    instruction: str,
    response: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant that answers questions accurately.",
) -> List[Dict[str, str]]:
    """
    Format instruction and response for chat template.

    Args:
        instruction: The user instruction/question
        response: The assistant response (optional, for training)
        system_prompt: System prompt to use

    Returns:
        List of message dictionaries for chat template
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]

    if response is not None:
        messages.append({"role": "assistant", "content": response})

    return messages


def convert_benchmark_to_llm_format(
    benchmark_name: str, df: pd.DataFrame, target_col: str, format_type: Literal["mc", "num"]
) -> tuple[List[TrainingDatum], str, str]:
    """
    Convert benchmark-specific data to LLM format with appropriate templates.

    Args:
        benchmark_name: Name of the benchmark (e.g., "video_mme", "vsi")
        df: DataFrame with benchmark data
        target_col: Target column name
        format_type: Format type

    Returns:
        Tuple of (training_data, instruction_template, response_template)
    """
    # Benchmark-specific templates
    templates = {
        "video_mme": {
            "instruction": "Based on the video content, answer the following question: {question}",
            "response": "The answer is {answer}.",
        },
        "vsi": {
            "instruction": "Looking at the spatial arrangement, answer: {question}",
            "response": "The answer is {answer}.",
        },
        "mmmu": {
            "instruction": "Based on the multimodal content, answer: {question}",
            "response": "The answer is {answer}.",
        },
        "cvb": {
            "instruction": "Analyze the visual content and answer: {question}",
            "response": "The answer is {answer}.",
        },
        "default": {"instruction": "Answer the following question: {question}", "response": "The answer is {answer}."},
    }

    # Get templates for this benchmark
    template_config = templates.get(benchmark_name, templates["default"])
    instruction_template = template_config["instruction"]
    response_template = template_config["response"]

    # Convert to training format
    training_data = convert_to_blind_training_format(
        df=df,
        target_col=target_col,
        format_type=format_type,
        instruction_template=instruction_template,
        response_template=response_template,
    )

    return training_data, instruction_template, response_template
