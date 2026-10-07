"""MMMU, VideoMME and MMStar adapters (TsT-LLM only). Offline tests use synthetic rows; the
benchmarks' own data are downloaded only by the network-marked tests."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from TsT.__main__ import main
from TsT.benchmarks.mmmu.benchmark import MMMUBenchmark
from TsT.benchmarks.mmstar.benchmark import MMStarBenchmark, parse_question
from TsT.benchmarks.video_mme.benchmark import VideoMMEBenchmark
from TsT.evaluators.llm.data.conversion import get_blind_qa


def test_llm_only():
    for cls in (MMMUBenchmark, VideoMMEBenchmark, MMStarBenchmark):
        bench = cls()
        assert bench.supports_llm and not bench.supports_rf
        with pytest.raises(NotImplementedError, match="TsT-RF is not available"):
            bench.get_feature_based_models()
        assert bench.get_metadata()["modes"] == ["llm"]
        assert len(bench.default_revision) == 40 and bench.default_revision in bench.row_order_fingerprints


def test_rf_mode_exits_2(capsys):
    assert main(["--benchmark", "mmstar"]) == 2
    assert "use --mode llm" in capsys.readouterr().err


def test_mmmu_preprocess():
    raw = pd.DataFrame(
        {
            "id": ["v_1", "v_2", "v_3"],
            "question": ["Q1 <image 1>", "Q2", "Q3"],
            "options": ["['x', 'y', 'z']", "['p', 'q']", "[]"],
            "answer": ["C", "A", "42"],
            "question_type": ["multiple-choice", "multiple-choice", "open"],
            "subfield": ["Optics"] * 3,
        }
    )
    df = MMMUBenchmark.preprocess(raw)
    assert df["question_format"].tolist() == ["mc", "mc", "oe"]
    assert df["gt_idx"].tolist() == [2, 0, -1] and df["gt_val"].tolist()[:2] == ["z", "p"]
    assert (df["question_type"] == "Other").all()  # subfields under 5 questions are merged
    assert [m.name for m in MMMUBenchmark().get_qa_models()] == ["mmmu_mc"]
    bad = raw.copy()
    bad.loc[0, "answer"] = "Z"
    with pytest.raises(ValueError, match="non-letter"):
        MMMUBenchmark.preprocess(bad)


def test_video_mme_preprocess_and_target():
    raw = pd.DataFrame(
        {
            "question_id": ["001-1", "001-2"],
            "videoID": ["v1", "v1"],
            "question": ["What happens?", "Who?"],
            "options": [np.array(["A. run", "B. sit", "C. fly", "D. swim"], dtype=object)] * 2,
            "answer": ["B", "D"],
            "task_type": ["Action Recognition", "Counting Problem"],
        }
    )
    df = VideoMMEBenchmark.preprocess(raw)
    assert df["gt_idx"].tolist() == [1, 3] and (df["question_format"] == "mc").all()
    (model,) = VideoMMEBenchmark().get_qa_models()
    assert model.default_target_col == "gt_idx"
    instruction, response, _, _ = get_blind_qa(df.iloc[0].to_dict(), "gt_idx", "mc")
    assert "Options:\nA. run\nB. sit\nC. fly\nD. swim\n" in instruction and response == "B"


@pytest.mark.parametrize(
    "raw,stem,options",
    [
        ("What is it?\nOptions: A: a cat, B: a dog, C: a bird", "What is it?", ["a cat", "a dog", "a bird"]),
        ("Which city? (A) Lyon (B) Springfield, D.C. (C) Turin", "Which city?", ["Lyon", "Springfield, D.C.", "Turin"]),
        ("Pick one.\nChoices: A: A red kite, B: a boat", "Pick one.", ["A red kite", "a boat"]),
        ("No options here", "No options here", None),
    ],
)
def test_mmstar_parse_question(raw, stem, options):
    assert parse_question(raw) == (stem, options)


def test_mmstar_preprocess_prelabels_options():
    raw = pd.DataFrame(
        {
            "index": [0, 1],
            "question": ["Q?\nOptions: A: A red kite, B: a boat", "R?\nOptions: A: x, B: y, C: z"],
            "answer": ["A", "C"],
            "category": ["coarse perception", "math"],
            "l2_category": ["c1", "c2"],
        }
    )
    df = MMStarBenchmark.preprocess(raw)
    assert df["options"].iloc[0] == ["A. A red kite", "B. a boat"]
    assert df["gt_val"].tolist() == ["A red kite", "z"] and df["gt_idx"].tolist() == [0, 2]
    instruction, _, _, _ = get_blind_qa(df.iloc[0].to_dict(), "ground_truth", "mc")
    assert "Options:\nA. A red kite\nB. a boat\n" in instruction
    bad = raw.copy()
    bad.loc[1, "question"] = "no options"
    with pytest.raises(ValueError, match="did not parse"):
        MMStarBenchmark.preprocess(bad)


def test_adapters_import_without_heavy_dependencies():
    code = (
        "import sys; from TsT.core.benchmark import BenchmarkRegistry; BenchmarkRegistry.list_benchmarks(); "
        "print(sorted(m for m in ('datasets', 'torch', 'vllm', 'sentence_transformers') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


@pytest.mark.network
@pytest.mark.parametrize(
    "cls,n_rows,formats",
    [
        (MMMUBenchmark, 900, {"mc": 847, "oe": 53}),
        (VideoMMEBenchmark, 2700, {"mc": 2700}),
        (MMStarBenchmark, 1500, {"mc": 1500}),
    ],
)
def test_pinned_data(cls, n_rows, formats):
    df = cls().load_data()  # checks the row-order fingerprint
    assert len(df) == n_rows and df[cls.id_col].is_unique
    assert df["question_format"].value_counts().to_dict() == formats
