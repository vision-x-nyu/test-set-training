"""
TsT-LLM fold evaluator.

For one question-answer model (one answer format of a benchmark), the evaluator first
scores every question with the base model (the zero-shot baseline). Then, for each fold,
it trains a LoRA adapter on the training folds with LLaMA-Factory and scores the
held-out fold with vLLM. A question's held-out score is s(x); s(x) minus its zero-shot
score is the per-question gain that IBP ranks by.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.protocols import BiasModel, EvaluationResult, FoldResult, ModelEvaluator
from .config import LLMRunConfig
from .data.conversion import convert_to_blind_test_instances, convert_to_blind_training_format
from .data.models import LLMPredictionResult, TestInstance
from .scoring import score_llm

logger = logging.getLogger(__name__)


def write_predictions(
    path: Path,
    tag: str,
    df: pd.DataFrame,
    id_col: Optional[str],
    test_instances: List[TestInstance],
    prediction_results: List[LLMPredictionResult],
    scores: List[float],
) -> None:
    """Write per-question predictions as JSON lines (one file per zero-shot pass or fold)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = df[id_col].tolist() if id_col and id_col in df.columns else [None] * len(df)
    with open(path, "w") as f:
        for rid, qid, inst, res, score in zip(df.index, ids, test_instances, prediction_results, scores):
            record = {
                "row_id": rid.item() if hasattr(rid, "item") else rid,
                "id": qid.item() if hasattr(qid, "item") else qid,
                "tag": tag,
                "instance_id": inst.instance_id,
                "ground_truth": inst.ground_truth,
                "options": inst.options,
                "prediction": res.prediction,
                "raw_output": res.raw_output,
                "option_probs": res.option_probs,
                "confidence": res.confidence,
                "score": float(score),
            }
            f.write(json.dumps(record, default=str) + "\n")


class LLMEvaluator(ModelEvaluator):
    """Zero-shot baseline plus k-fold LoRA fine-tuning for one question-answer model."""

    def __init__(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        llm_config: Optional[LLMRunConfig] = None,
        seed: int = 42,
        predictions_dir: Optional[Path] = None,
        id_col: Optional[str] = None,
    ):
        from .predictors.vllm import VLLMPredictor

        self.model = model
        self.qdf = model.select_rows(df)
        self.target_col = target_col
        self.llm_config = llm_config or LLMRunConfig()
        self.seed = seed
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else None
        self.id_col = id_col
        self.predictor = VLLMPredictor(self.llm_config.to_predictor_config())
        self.zero_shot_baseline, self.zero_shot_sample_scores = self._evaluate_zero_shot()

    def _predictions_path(self, tag: str) -> Optional[Path]:
        return None if self.predictions_dir is None else self.predictions_dir / f"{self.model.name}_{tag}.jsonl"

    def _score(self, df: pd.DataFrame, id_prefix: str, seed: int, tag: str) -> List[float]:
        instances = convert_to_blind_test_instances(df, self.target_col, self.model.format, id_prefix=id_prefix)
        results = self.predictor.predict(instances)
        if instances:
            logger.info(f"First prompt:\n{instances[0].instruction}")
            logger.info(f"First ground truth: {instances[0].ground_truth}; prediction: {results[0].raw_output!r}")
        scores = [score_llm(res, inst, self.model.format, seed) for res, inst in zip(results, instances)]
        path = self._predictions_path(tag)
        if path is not None:
            write_predictions(path, tag, df, self.id_col, instances, results, scores)
        return scores

    def _evaluate_zero_shot(self) -> Tuple[float, Dict[int, float]]:
        logger.info(f"[{self.model.name}] zero-shot pass with {self.llm_config.model_name}")
        scores = self._score(self.qdf, "zero_shot", self.seed, "zero_shot")
        self.predictor.reset()  # free GPU memory for LoRA training
        sample_scores = dict(zip(self.qdf.index, scores))
        baseline = float(np.mean(scores)) if scores else float("nan")
        logger.info(f"[{self.model.name}] zero-shot score: {baseline:.2%} ({len(scores)} questions)")
        return baseline, sample_scores

    def train_and_evaluate_fold(
        self,
        model: BiasModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        from .trainers.llamafactory import LlamaFactoryTrainer

        train_data = convert_to_blind_training_format(train_df, target_col, self.model.format)
        trainer = LlamaFactoryTrainer(self.llm_config.to_trainer_config(seed=seed))
        # The adapter lives in a temporary directory, deleted after the fold is scored.
        with tempfile.TemporaryDirectory(prefix=f"tst_fold{fold_id}_") as tmp:
            logger.info(f"[{self.model.name}] fold {fold_id}: training a LoRA adapter on {len(train_data)} questions")
            self.predictor.reset()
            try:
                adapter = trainer.train(train_data, Path(tmp))
                self.predictor.ensure_loaded()
                self.predictor.load_adapter(str(adapter.adapter_path))
                scores = self._score(test_df, f"fold_{fold_id}", seed, f"seed{seed}_fold{fold_id}")
            finally:
                self.predictor.reset()  # free GPU memory for the next fold or model

        fold_score = float(np.mean(scores))
        logger.info(
            f"[{self.model.name}] fold {fold_id}: {fold_score:.2%} "
            f"(zero-shot on all questions: {self.zero_shot_baseline:.2%})"
        )
        return FoldResult(
            fold_id=fold_id,
            score=fold_score,
            train_size=len(train_df),
            test_idx=list(test_df.index),
            metric=self.model.metric,
            metadata={"model_name": self.llm_config.model_name},
            sample_predictions=dict(zip(test_df.index, scores)),
        )

    def process_results(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        evaluation_result.feature_importances = None
        evaluation_result.zero_shot_baseline = self.zero_shot_baseline
        evaluation_result.zero_shot_sample_scores = dict(self.zero_shot_sample_scores)
        evaluation_result.model_metadata.update(
            {
                "zero_shot_baseline": self.zero_shot_baseline,
                "improvement": evaluation_result.overall_mean - self.zero_shot_baseline,
            }
        )
        return evaluation_result
