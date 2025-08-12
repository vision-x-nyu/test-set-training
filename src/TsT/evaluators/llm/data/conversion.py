"""
Dataset format conversion utilities for TsT LLM training.

This module provides functions to convert between different data formats
used in the TsT framework and LLM training pipelines.
"""

import pandas as pd
from typing import List, Literal, Dict, Optional, Any
from .models import TrainingDatum, TestInstance


def get_blind_qa(
    record: Dict[str, Any],
    target_col: str,
    format_type: Literal["mc", "num", "oe"],
    instruction_template: str = "Answer the following question: {question}",
    post_prompt: str = "Answer with the option's letter from the given choices directly.",
    response_template: str = "{answer}",
):
    """
    Convert a single record to blind QA format.
    """

    assert "{question}" in instruction_template, (
        f"instruction_template must contain {{question}}, got {instruction_template}"
    )
    assert "{answer}" in response_template, f"response_template must contain {{answer}}, got {response_template}"

    # Extract question text - try different common column names
    question_text = None
    for col in ["question", "text", "instruction", "query"]:
        if col in record and pd.notna(record[col]):
            question_text = str(record[col])
            break

    if question_text is None:
        raise ValueError("No question text found in the row")

    # get instruction and answer
    target = record[target_col]
    match format_type:
        case "mc":
            # Include answer choices in the question
            if "choices" in record:
                options = record["choices"]
            elif "options" in record:
                options = record["options"]
            else:
                raise ValueError(f"No choices found in the row: {record}")

            options_text = "\n".join(options)
            question_text = f"{question_text} Options:\n{options_text}"
            if target_col == "gt_idx" and isinstance(target, int):
                # Convert index to letter
                answer = chr(65 + int(target))
            else:
                answer = str(target)
        case "num" | "oe":
            answer = str(target)
            options = None
        case _:
            raise ValueError(f"Invalid format type: {format_type}")

    instruction = instruction_template.format(question=question_text)
    instruction += "\n" + post_prompt if post_prompt else ""

    response = response_template.format(answer=answer)

    return instruction, response, answer, options


def convert_to_blind_training_format(
    df: pd.DataFrame,
    target_col: str,
    format_type: Literal["mc", "num", "oe"],
    instruction_template: str = "Answer the following question: {question}",
    post_prompt: str = "Answer with the option's letter from the given choices directly.",
    response_template: str = "{answer}",
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
        instruction, response, answer, options = get_blind_qa(
            row, target_col, format_type, instruction_template, post_prompt, response_template
        )

        training_data.append(
            TrainingDatum(
                instruction=instruction,
                response=response,
                metadata={
                    "row_id": idx,
                    "format_type": format_type,
                    "target_col": target_col,
                    "original_answer": answer,
                    "options": options,
                },
            )
        )

    return training_data


def convert_to_blind_test_instances(
    df: pd.DataFrame,
    target_col: str,
    format_type: Literal["mc", "num", "oe"],
    instruction_template: str = "Answer the following question: {question}",
    post_prompt: str = "Answer with the option's letter from the given choices directly.",
    response_template: str = "{answer}",
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
        instruction, response, answer, options = get_blind_qa(
            row, target_col, format_type, instruction_template, post_prompt, response_template
        )

        instance_id = f"{id_prefix}_{idx}"

        test_instances.append(
            TestInstance(
                instance_id=instance_id,
                instruction=instruction,
                ground_truth=response,
                options=options,
                metadata={
                    "row_id": idx,
                    "format_type": format_type,
                    "target_col": target_col,
                    "original_answer": answer,
                },
            )
        )

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
