"""
Scoring TsT-LLM predictions.

- Multiple choice: the probability of the gold option, from the log-probabilities of the
  first generated token over the option letters (normalized over the letters found). If
  the gold letter is not among the top log-probabilities, the generated text is parsed
  instead and scored 1 if it names the gold option, else 0. Text with no recognizable
  option gets a random option, as in the MMMU evaluation code; that choice is seeded by
  the run seed and the question, so this choice no longer varies between reruns. (The paper's
  runs drew it unseeded, which moved zero-shot scores by up to a few tenths of a point.)
- Numerical: mean relative accuracy of the generated number; an answer with no number
  scores 0.
- Open-ended: normalized exact match.
"""

import math
import random
from typing import Dict, List, Literal, Optional

import numpy as np

from ...utils import fuzzy_match, fuzzy_mra
from .data.models import LLMPredictionResult, TestInstance


def get_multi_choice_info(options):
    """Map option letters to option texts.

    Adapted from the MMMU evaluation code (Apache-2.0):
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """
    all_choices = [chr(ord("A") + i) for i in range(len(options))]
    index2ans = dict(zip(all_choices, options))
    return index2ans, all_choices


def parse_multi_choice_response(response: str, options: List[str], rng: random.Random) -> str:
    """The option letter named in ``response``; a random letter (drawn from ``rng``) if none is found.

    Adapted from the MMMU evaluation code (Apache-2.0):
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    index2ans, all_choices = get_multi_choice_info(options)
    candidate = _find_choice(response, all_choices, index2ans)
    return candidate if candidate is not None else rng.choice(all_choices)


def gold_letter(ground_truth: str, options: List[str]) -> str:
    """The gold option letter. Raises if ``ground_truth`` does not name exactly one option."""
    index2ans, all_choices = get_multi_choice_info(options)
    gt = ground_truth.strip()
    if gt in all_choices:
        return gt
    candidate = _find_choice(gt, all_choices, index2ans)
    if candidate is None:
        raise ValueError(f"Gold answer {ground_truth!r} does not name one of the options {all_choices}")
    return candidate


def _find_choice(response: str, all_choices: List[str], index2ans: Dict[str, str]) -> Optional[str]:
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # no letter found: for longer responses, look for an option's text
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # several candidates: take the last one mentioned
    start_indexes = []
    if index_ans:
        if ans_with_brack:
            for can in candidates:
                start_indexes.append(response.rfind(f"({can})"))
        else:
            for can in candidates:
                start_indexes.append(response.rfind(f" {can} "))
    else:
        for can in candidates:
            start_indexes.append(response.lower().rfind(index2ans[can].lower()))
    return candidates[int(np.argmax(start_indexes))]


def option_token_ids(tokenizer, options: List[str]) -> Dict[str, List[int]]:
    """Candidate token ids for each option letter: "A" and " A" (tokenizers differ)."""
    option_tokens: Dict[str, List[int]] = {}
    for i, _opt in enumerate(options):
        letter = chr(65 + i)
        candidate_ids = set()
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            candidate_ids.add(ids[0])
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            candidate_ids.add(ids_space[-1])  # the letter token
        option_tokens[letter] = list(candidate_ids)
    return option_tokens


def option_probs_from_logprobs(
    logprobs: Dict[int, float], option_tokens: Dict[str, List[int]]
) -> Optional[Dict[str, float]]:
    """Normalized probabilities of the option letters among the returned top log-probabilities.

    ``logprobs`` maps token id to log-probability. Each letter takes its most probable
    token variant. Letters absent from ``logprobs`` are left out; None if no letter is present.
    """
    raw_probs: Dict[str, float] = {}
    for letter, candidate_ids in option_tokens.items():
        found = [math.exp(logprobs[tid]) for tid in candidate_ids if tid in logprobs]
        if found:
            raw_probs[letter] = max(found)
    if not raw_probs:
        return None
    total = sum(raw_probs.values())
    if total <= 0:
        return None
    return {letter: prob / total for letter, prob in raw_probs.items()}


def score_llm(
    result: LLMPredictionResult,
    test_instance: TestInstance,
    format_type: Literal["mc", "num", "oe"],
    seed: int,
) -> float:
    """Score one prediction (see the module docstring). ``seed`` seeds the MC parsing fallback."""
    match format_type:
        case "mc":
            if result.confidence is not None:
                return float(result.confidence)
            rng = random.Random(f"{seed}:{test_instance.instance_id}")
            pred = parse_multi_choice_response(result.prediction, test_instance.options, rng)
            return float(pred == gold_letter(test_instance.ground_truth, test_instance.options))
        case "num":
            score = fuzzy_mra(result.prediction, test_instance.ground_truth)
            return 0.0 if np.isnan(score) else float(score)
        case "oe":
            return fuzzy_match(result.prediction, test_instance.ground_truth)
        case _:
            raise ValueError(f"Unknown format_type: {format_type}")
