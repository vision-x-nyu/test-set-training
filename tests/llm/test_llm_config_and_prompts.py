"""TsT-LLM settings, training config and blind prompts (CPU; no torch or vLLM needed)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from TsT.evaluators.llm import LLMRunConfig
from TsT.evaluators.llm.data.conversion import convert_to_blind_training_format, get_blind_qa
from TsT.evaluators.llm.trainers.llamafactory import single_gpu_env, training_config, write_alpaca_dataset

REPO = Path(__file__).resolve().parents[2]


def test_defaults_are_app_d1():
    c = LLMRunConfig()
    assert c.model_name == "Qwen/Qwen2-7B-Instruct"
    assert (c.lora_rank, c.lora_alpha, c.lora_dropout, c.num_epochs) == (8, 16, 0.1, 1)
    assert (c.train_batch_size, c.eval_batch_size, c.learning_rate, c.max_seq_length) == (4, 4, 2e-4, 1024)
    assert (c.template, c.apply_chat_template, c.temperature, c.max_tokens, c.logprob_top_k) == (
        "gemma",
        False,
        0.0,
        10,
        100,
    )


def test_training_config_is_the_published_yaml():
    cfg = training_config(
        LLMRunConfig().to_trainer_config(seed=42), "<fold_tmp>/dataset", "custom_sft_dataset", "<fold_tmp>/adapter"
    )
    reference = REPO / "configs" / "app_d1" / "llamafactory_train.yaml"
    body = "".join(line for line in reference.read_text().splitlines(True) if not line.startswith("#"))
    assert yaml.dump(cfg, default_flow_style=False) == body


def test_model_revision_and_seed_reach_the_training_config():
    cfg = training_config(LLMRunConfig(model_revision="abc").to_trainer_config(seed=43), "d", "custom_sft_dataset", "o")
    assert cfg["model_revision"] == "abc" and cfg["seed"] == 43


def test_alpaca_dataset_files(tmp_path):
    from TsT.evaluators.llm.data.models import TrainingDatum

    name = write_alpaca_dataset([TrainingDatum("q1", "A"), TrainingDatum("q2", "3")], tmp_path)
    info = (tmp_path / "dataset_info.json").read_text()
    assert info == (
        '{"custom_sft_dataset": {"file_name": "custom_sft_dataset.jsonl", "formatting": "alpaca", '
        '"ranking": false, "load_from": "file", "columns": {"prompt": "instruction", "response": "output"}}}'
    )
    lines = (tmp_path / "custom_sft_dataset.jsonl").read_text().splitlines()
    assert name == "custom_sft_dataset" and lines == [
        '{"instruction": "q1", "output": "A"}',
        '{"instruction": "q2", "output": "3"}',
    ]


def test_single_gpu_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert single_gpu_env()["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert single_gpu_env()["CUDA_VISIBLE_DEVICES"] == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert single_gpu_env()["CUDA_VISIBLE_DEVICES"] == "2"


def test_mc_prompt_vsi():
    record = {
        "question": "Which is closest to the sofa?",
        "options": np.array(["A. bed", "B. lamp"], dtype=object),
        "ground_truth": "B",
    }
    instruction, response, answer, options = get_blind_qa(record, "ground_truth", "mc")
    assert instruction == (
        "Answer the following question: Which is closest to the sofa? Options:\nA. bed\nB. lamp\n"
        "Answer with the option's letter from the given choices directly."
    )
    assert (response, answer, options) == ("B", "B", ["A. bed", "B. lamp"])


def test_mc_prompt_gt_idx_target_and_labelling():
    # CV-Bench: unlabelled choices get letters; a gt_idx target becomes the letter
    record = {"question": "How many cars?", "choices": ["0", "2", "1"], "gt_idx": 1}
    instruction, response, _, _ = get_blind_qa(record, "gt_idx", "mc")
    assert "Options:\nA. 0\nB. 2\nC. 1\n" in instruction and response == "B"
    # an option that already starts with its own letter and a space is used as is (as run in the paper)
    record = {"question": "Q?", "options": ["A tract carrying fibers", "nerves"], "ground_truth": "B"}
    instruction, _, _, _ = get_blind_qa(record, "ground_truth", "mc")
    assert "Options:\nA tract carrying fibers\nB. nerves\n" in instruction


def test_num_prompt():
    instruction, response, _, options = get_blind_qa(
        {"question": "How many chairs?", "ground_truth": "4"}, "ground_truth", "num"
    )
    assert instruction == "Answer the following question: How many chairs?\nAnswer with just a number."
    assert response == "4" and options is None


def test_training_examples_keep_row_ids(vsi_df):
    mc = vsi_df[vsi_df["question_format"] == "mc"].head(3)
    data = convert_to_blind_training_format(mc, "ground_truth", "mc")
    assert [d.metadata["row_id"] for d in data] == list(mc.index)
    assert all(d.response in "ABCD" for d in data)


def test_llm_package_imports_without_torch_or_vllm():
    code = (
        "import sys, TsT.evaluators.llm as l, TsT.evaluators.llm.scoring, TsT.evaluators.llm.trainers.llamafactory; "
        "l.LLMRunConfig(); print('torch' in sys.modules or 'vllm' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_preflight_names_the_extra_and_never_imports_torch():
    code = (
        "import sys; from TsT.evaluators.llm import missing_requirements; m = missing_requirements(); "
        "print('torch' in sys.modules); print(m)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0] == "False"  # importing torch here could initialize CUDA before vLLM forks
    import importlib.util

    if importlib.util.find_spec("vllm") is None:
        assert "uv sync --frozen --extra llm" in out[1]
    else:  # the llm extra is installed
        assert out[1] == "None"
