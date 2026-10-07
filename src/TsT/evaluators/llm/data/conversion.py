"""
Blind (text-only) prompts for TsT-LLM training and inference.

Each question becomes "Answer the following question: <question>", followed for
multiple-choice questions by " Options:" and one lettered option per line, and then a
format-specific instruction. No image or video is ever included.
"""

import ast
import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from ....utils import is_integer
from .models import TestInstance, TrainingDatum

logger = logging.getLogger(__name__)

MC_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NUM_POST_PROMPT = "Answer with just a number."
OE_POST_PROMPT = "Answer concisely."

DEFAULT_POST_PROMPTS: Dict[str, str] = {
    "mc": MC_POST_PROMPT,
    "num": NUM_POST_PROMPT,
    "oe": OE_POST_PROMPT,
}
DEFAULT_INSTRUCTION_TEMPLATE = "Answer the following question: {question}"


def get_blind_qa(
    record: Dict[str, Any],
    target_col: str,
    format_type: Literal["mc", "num", "oe"],
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    post_prompt: Optional[str] = None,
    response_template: str = "{answer}",
):
    """Blind prompt for one record. Returns (instruction, response, answer, options).

    Options get letter labels ("A. ...") unless an option's text already starts with its
    own letter followed by "." or a space; such options are used as they are (this is how
    the paper's runs rendered them). A ``gt_idx`` target becomes the option letter.
    """
    if "{question}" not in instruction_template:
        raise ValueError(f"instruction_template must contain {{question}}, got {instruction_template}")
    if "{answer}" not in response_template:
        raise ValueError(f"response_template must contain {{answer}}, got {response_template}")

    question_text = None
    for col in ["question", "text", "instruction", "query"]:
        if col in record and pd.notna(record[col]):
            question_text = str(record[col])
            break
    if question_text is None:
        raise ValueError("No question text found in the row")

    target = record[target_col]
    match format_type:
        case "mc":
            if "choices" in record:
                options = record["choices"]
            elif "options" in record:
                options = record["options"]
            else:
                options = None
            if isinstance(options, str):
                options = ast.literal_eval(options)
            if not isinstance(options, Iterable) or len(options) == 0:
                raise ValueError(f"No valid choices/options found in MC row: {record}")

            options = list(options)
            labeled = []
            for i, opt in enumerate(options):
                letter = chr(65 + i)
                if not str(opt).strip().startswith(f"{letter}.") and not str(opt).strip().startswith(f"{letter} "):
                    labeled.append(f"{letter}. {opt}")
                else:
                    labeled.append(str(opt))
            options_text = "\n".join(labeled)
            question_text = f"{question_text} Options:\n{options_text}"

            if target_col == "gt_idx" and is_integer(target):
                answer = chr(65 + int(target))
            else:
                answer = str(target)
        case "num" | "oe":
            answer = str(target)
            options = None
        case _:
            raise ValueError(f"Invalid format type: {format_type}")

    instruction = instruction_template.format(question=question_text)
    resolved_post_prompt = post_prompt if post_prompt is not None else DEFAULT_POST_PROMPTS.get(format_type, "")
    if resolved_post_prompt:
        instruction += "\n" + resolved_post_prompt

    response = response_template.format(answer=answer)
    return instruction, response, answer, options


def convert_to_blind_training_format(
    df: pd.DataFrame,
    target_col: str,
    format_type: Literal["mc", "num", "oe"],
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    post_prompt: Optional[str] = None,
    response_template: str = "{answer}",
) -> List[TrainingDatum]:
    """Fine-tuning examples for the rows of ``df``."""
    training_data = []
    for idx, row in df.iterrows():
        instruction, response, answer, options = get_blind_qa(
            row.to_dict(), target_col, format_type, instruction_template, post_prompt, response_template
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
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    post_prompt: Optional[str] = None,
    response_template: str = "{answer}",
    id_prefix: str = "test",
) -> List[TestInstance]:
    """Test instances for the rows of ``df`` (instance ids are ``{id_prefix}_{row index}``)."""
    test_instances = []
    for idx, row in df.iterrows():
        instruction, response, answer, options = get_blind_qa(
            row.to_dict(), target_col, format_type, instruction_template, post_prompt, response_template
        )
        test_instances.append(
            TestInstance(
                instance_id=f"{id_prefix}_{idx}",
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
