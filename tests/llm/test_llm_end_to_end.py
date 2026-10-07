"""TsT-LLM end to end on the offline fixtures, with a fake predictor and trainer (no GPU)."""

import json

import pytest

import TsT.evaluators.llm.predictors.vllm as vllm_module
import TsT.evaluators.llm.trainers.llamafactory as lf_module
from TsT.evaluation import evaluate_benchmark
from TsT.evaluators.llm.data.models import LLMPredictionResult, LoRAAdapterInfo


class FakePredictor:
    """Zero-shot: P(gold) = 0.25 for MC, answers "0" for NUM. With an adapter: P(gold) = 0.75, exact NUM."""

    instances = []

    def __init__(self, config, enforce_eager=False):
        self.config, self.adapter, self.loaded = config, None, True

    is_loaded = property(lambda self: self.loaded)
    current_adapter_path = property(lambda self: self.adapter)

    def ensure_loaded(self):
        self.loaded = True

    def reset(self):
        self.loaded, self.adapter = False, None

    def load_adapter(self, path):
        assert self.loaded, "base model must be loaded before an adapter"
        self.adapter = path

    def predict(self, instances):
        assert self.loaded
        out = []
        for inst in instances:
            if inst.options:
                out.append(LLMPredictionResult(inst.instance_id, "A", confidence=0.75 if self.adapter else 0.25))
            else:
                out.append(LLMPredictionResult(inst.instance_id, inst.ground_truth if self.adapter else "0"))
        return out


class FakeTrainer:
    seeds = []

    def __init__(self, config):
        self.config = config

    def train(self, training_data, output_dir):
        FakeTrainer.seeds.append(self.config.seed)
        (output_dir / "adapter").mkdir()
        return LoRAAdapterInfo(output_dir / "adapter", len(training_data), self.config.model_name)


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(vllm_module, "VLLMPredictor", FakePredictor)
    monkeypatch.setattr(lf_module, "LlamaFactoryTrainer", FakeTrainer)
    FakeTrainer.seeds = []


def test_llm_run_summary_s_x_and_predictions(fake_llm, vsi_df, tmp_path):
    run = evaluate_benchmark(
        "vsi", mode="llm", df=vsi_df, n_splits=3, predictions_dir=tmp_path / "pred", show_progress=False
    )
    s = run.summary()
    assert s["mode"] == "llm" and s["feature_set"] is None and s["llm_config"]["model_name"] == "Qwen/Qwen2-7B-Instruct"
    by_model = {m["question_type"]: m for m in s["per_question_type"]}
    assert by_model["vsi_mc"]["zero_shot"] == pytest.approx(0.25) and by_model["vsi_mc"]["score"] == pytest.approx(0.75)
    assert by_model["vsi_num"]["zero_shot"] == pytest.approx(0.0) and by_model["vsi_num"]["score"] == pytest.approx(1.0)
    n_mc, n_num = by_model["vsi_mc"]["count"], by_model["vsi_num"]["count"]
    assert n_mc + n_num == len(vsi_df)
    assert s["zero_shot_weighted_mean"] == pytest.approx(0.25 * n_mc / len(vsi_df))
    assert s["delta_weighted"] == pytest.approx(s["weighted_mean"] - s["zero_shot_weighted_mean"])
    assert FakeTrainer.seeds == [42, 42, 42, 42, 42, 42]  # 2 models x 3 folds, repeat 0

    sx = run.s_x()
    assert len(sx) == len(vsi_df) and sx["id"].is_unique
    mc = sx[sx["format"] == "mc"]
    assert (mc["s"] == 0.75).all() and (mc["s_zero_shot"] == 0.25).all() and (mc["delta"] == 0.5).all()

    files = sorted(p.name for p in (tmp_path / "pred").iterdir())
    assert files == sorted(
        [f"vsi_{f}_zero_shot.jsonl" for f in ("mc", "num")]
        + [f"vsi_{f}_seed42_fold{k}.jsonl" for f in ("mc", "num") for k in (1, 2, 3)]
    )
    rec = json.loads((tmp_path / "pred" / "vsi_mc_seed42_fold1.jsonl").read_text().splitlines()[0])
    assert rec["id"] in set(vsi_df["id"]) and rec["score"] == 0.75 and rec["tag"] == "seed42_fold1"
    pooled = [
        json.loads(line)["score"]
        for k in (1, 2, 3)
        for line in (tmp_path / "pred" / f"vsi_num_seed42_fold{k}.jsonl").read_text().splitlines()
    ]
    assert sum(pooled) / len(pooled) == pytest.approx(by_model["vsi_num"]["score"])


def test_llm_mode_rejects_rf_only_options(fake_llm, vsi_df):
    with pytest.raises(ValueError, match="TsT-RF only"):
        evaluate_benchmark("vsi", mode="llm", df=vsi_df, feature_set="paper")


def test_grouped_folds_in_llm_mode(fake_llm, vsi_df):
    run = evaluate_benchmark("vsi", mode="llm", df=vsi_df, n_splits=3, group_col="scene_name", show_progress=False)
    for res in run.results:
        for fold in res.repeat_results[0].fold_results:
            test_scenes = set(vsi_df.loc[fold.test_idx, "scene_name"])
            train_idx = set(res.zero_shot_sample_scores) - set(fold.test_idx)
            assert test_scenes.isdisjoint(set(vsi_df.loc[sorted(train_idx), "scene_name"]))


def test_cvb_llm_target_is_option_letter(fake_llm, cvb_df, tmp_path):
    run = evaluate_benchmark("cvb", mode="llm", df=cvb_df, n_splits=2, predictions_dir=tmp_path, show_progress=False)
    rec = json.loads((tmp_path / "cvb_mc_zero_shot.jsonl").read_text().splitlines()[0])
    assert rec["ground_truth"] in "ABCDEF" and run.summary()["per_question_type"][0]["target_col"] == "gt_idx"


def test_summarizer_matches_run(fake_llm, vsi_df, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "summarize_llm_predictions", Path(__file__).resolve().parents[2] / "scripts" / "summarize_llm_predictions.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = evaluate_benchmark("vsi", mode="llm", df=vsi_df, n_splits=2, predictions_dir=tmp_path, show_progress=False)
    summary = mod.summarize(tmp_path)
    for res in run.results:
        assert summary[res.model_name]["tst_pooled"]["score"] == pytest.approx(res.overall_mean)
        assert summary[res.model_name]["zero_shot"]["score"] == pytest.approx(res.zero_shot_baseline)
    assert summary["vsi_mc"]["tst_pooled"]["top1_acc"] == pytest.approx(
        (vsi_df.loc[vsi_df["question_format"] == "mc", "ground_truth"] == "A").mean()
    )


def test_ibp_per_format_skips_unscored_formats(fake_llm):
    """MMMU-like data: TsT-LLM scores only the MC questions, so per-format IBP prunes only 'mc'."""
    import pandas as pd

    from TsT.benchmarks.mmmu.benchmark import MMMUBenchmark
    from TsT.debiasing import debias_benchmark

    n_mc, n_oe = 24, 6
    raw = pd.DataFrame(
        {
            "id": [f"q{i:02d}" for i in range(n_mc + n_oe)],
            "question": [f"Question {i}?" for i in range(n_mc + n_oe)],
            "options": ["['a', 'b', 'c']"] * n_mc + ["[]"] * n_oe,
            "answer": ["ABC"[i % 3] for i in range(n_mc)] + ["7"] * n_oe,
            "question_type": ["multiple-choice"] * n_mc + ["open"] * n_oe,
            "subfield": ["Optics"] * (n_mc + n_oe),
        }
    )
    df = MMMUBenchmark.preprocess(raw)
    run = debias_benchmark("mmmu", mode="llm", alloc="per_format", budget=4, batch_size=2, n_splits=2, df=df)
    groups = {g["group"]: g["budget"] for g in run.summary()["groups"]}
    assert groups == {"mc": 4} and run.n_removed == 4
    assert set(run.removed_ids) <= set(raw["id"][:n_mc])
