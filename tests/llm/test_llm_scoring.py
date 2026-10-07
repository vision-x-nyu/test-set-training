"""TsT-LLM scoring rules."""

import math
import random

import pytest

from TsT.evaluators.llm.data.models import LLMPredictionResult, TestInstance
from TsT.evaluators.llm.scoring import (
    gold_letter,
    option_probs_from_logprobs,
    parse_multi_choice_response,
    score_llm,
)

OPTIONS = ["A. red", "B. blue", "C. green", "D. black"]


def _mc(pred, confidence=None, iid="fold_1_7"):
    return LLMPredictionResult(iid, pred, confidence=confidence), TestInstance(iid, "q", "C", options=OPTIONS)


def test_mc_score_is_gold_probability_when_available():
    res, inst = _mc("A", confidence=0.31)
    assert score_llm(res, inst, "mc", seed=42) == 0.31


def test_mc_fallback_parses_text():
    assert score_llm(*_mc("C."), "mc", seed=42) == 1.0
    assert score_llm(*_mc("(B)"), "mc", seed=42) == 0.0


def test_mc_random_fallback_is_seeded_per_question():
    scores = {score_llm(*_mc("I cannot tell", iid=f"zero_shot_{i}"), "mc", seed=42) for i in range(40)}
    assert scores == {0.0, 1.0}  # a random option, so sometimes right
    again = [score_llm(*_mc("I cannot tell", iid=f"zero_shot_{i}"), "mc", seed=42) for i in range(40)]
    first = [score_llm(*_mc("I cannot tell", iid=f"zero_shot_{i}"), "mc", seed=42) for i in range(40)]
    assert again == first  # identical on rerun, whatever else was scored before
    other_seed = [score_llm(*_mc("I cannot tell", iid=f"zero_shot_{i}"), "mc", seed=7) for i in range(40)]
    assert other_seed != first


def test_parser_requires_an_rng_for_the_fallback():
    assert parse_multi_choice_response("B", OPTIONS, random.Random(0)) == "B"
    with pytest.raises(TypeError):
        parse_multi_choice_response("no letter here", OPTIONS)  # type: ignore[call-arg]


def test_gold_letter_is_strict():
    assert gold_letter("D", OPTIONS) == "D"
    with pytest.raises(ValueError):
        gold_letter("nothing", OPTIONS)


def test_num_unparseable_scores_zero():
    inst = TestInstance("i", "q", "4")
    assert score_llm(LLMPredictionResult("i", "4"), inst, "num", seed=0) == 1.0
    assert score_llm(LLMPredictionResult("i", "four"), inst, "num", seed=0) == 1.0
    assert score_llm(LLMPredictionResult("i", "no idea"), inst, "num", seed=0) == 0.0


def test_option_probs_normalized_over_found_letters():
    tokens = {"A": [10, 110], "B": [11], "C": [12]}
    probs = option_probs_from_logprobs(
        {10: math.log(0.2), 110: math.log(0.4), 11: math.log(0.2), 99: math.log(0.1)}, tokens
    )
    assert probs == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    assert option_probs_from_logprobs({99: 0.0}, tokens) is None
