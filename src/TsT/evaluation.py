"""
Run TsT evaluations and summarize their results.

- run_evaluation: k-fold cross-validation for a list of bias models on one dataframe.
- evaluate_benchmark: load a registered benchmark at a pinned revision and run TsT-RF
  (mode="rf") or TsT-LLM (mode="llm") on it.
- summarize_results / per_question_scores: overall statistics and the per-question s(x) table.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .core.cross_validation import CrossValidationConfig, UnifiedCrossValidator
from .core.protocols import BiasModel, EvaluationResult, FoldResult, ModelEvaluator, RepeatResult
from .utils import row_order_sha1, weighted_mean_std

logger = logging.getLogger(__name__)


class ModelEvaluationError(RuntimeError):
    """Raised when a model fails during evaluation (unless keep_going=True)."""


# Versions that produced the published TsT-RF values (the pins in uv.lock and constraints.txt).
# Other versions run, but can move the headline score by about a point and per-type
# scores by several points.
LOCKED_VERSIONS = {"numpy": "1.26.4", "pandas": "2.3.0", "scikit-learn": "1.6.1"}


def version_mismatches() -> Dict[str, Dict[str, Optional[str]]]:
    """{package: {"installed": ..., "locked": ...}} for LOCKED_VERSIONS packages that differ."""
    from importlib.metadata import PackageNotFoundError, version

    out: Dict[str, Dict[str, Optional[str]]] = {}
    for pkg, locked in LOCKED_VERSIONS.items():
        try:
            installed: Optional[str] = version(pkg)
        except PackageNotFoundError:
            installed = None
        if installed != locked:
            out[pkg] = {"installed": installed, "locked": locked}
    return out


def resolve_question_types(models: Sequence[BiasModel], requested: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Map requested question types to model names, in model order (None means all).

    Accepts model names (the "question type" column of the results table) and, as aliases,
    the dataset question types that one model scores together (a model's optional
    ``aliases`` attribute, e.g. VSI-Bench's object_rel_direction_easy/medium/hard for the
    object_rel_direction model). Raises ValueError naming every unknown entry and listing
    the valid names.
    """
    if requested is None:
        return None
    names = [m.name for m in models]
    alias_of: Dict[str, str] = {}
    for m in models:
        for alias in getattr(m, "aliases", ()):
            alias_of.setdefault(alias, m.name)
    wanted: set = set()
    via_alias: Dict[str, List[str]] = {}
    unknown: List[str] = []
    for q in (str(x).strip() for x in requested):
        if q in names:
            wanted.add(q)
        elif q in alias_of:
            wanted.add(alias_of[q])
            via_alias.setdefault(alias_of[q], []).append(q)
        else:
            unknown.append(q if q else "''")
    if unknown or not wanted:
        msg = f"Unknown question type(s): {', '.join(unknown) or '(none given)'}. Valid names: {', '.join(names)}"
        if alias_of:
            aliases = ", ".join(f"{a} (-> {n})" for a, n in alias_of.items())
            msg += f". Dataset question types accepted as aliases: {aliases}"
        raise ValueError(msg)
    for name, used in via_alias.items():
        model = next(m for m in models if m.name == name)
        logger.warning(
            f"{', '.join(used)}: scored by the '{name}' model, which evaluates all of "
            f"{', '.join(getattr(model, 'aliases', ()))} together"
        )
    return [n for n in names if n in wanted]


# =============================================================================
# UNIFIED EVALUATION -----------------------------------------------------------
# =============================================================================


def run_evaluation(
    question_models: List[BiasModel],
    df_full: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False,
    repeats: int = 1,
    question_types: Union[List[str], None] = None,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    fold_seed: Optional[int] = None,
    keep_going: bool = False,
    evaluator: Optional[ModelEvaluator] = None,
    show_progress: bool = True,
    evaluator_factory: Optional[Callable[[BiasModel, pd.DataFrame, str], ModelEvaluator]] = None,
) -> List[EvaluationResult]:
    """
    Run k-fold evaluation for all models and return one result per model.

    Args:
        question_models: Bias models to evaluate (one per question type).
        df_full: The full dataframe containing all data.
        n_splits: Number of cross-validation folds (k).
        random_state: Seed for fold assignment and the Random Forest (random_state + repeat index per repeat).
        verbose: Log per-model details (fold scores, feature importances).
        repeats: Number of repeats with different seeds (random_state + repeat index).
        question_types: Optional list of model names to evaluate (see resolve_question_types;
            unknown names raise ValueError). If None, evaluate all.
        target_col: Target column. None uses each model's default target (see
            UnifiedCrossValidator.resolve_target_column).
        group_col: Optional column whose values define CV groups (grouped folds).
        fold_seed: Optional seed for the fold partition only (fold_seed + repeat index);
            None uses random_state. The Random Forest always uses random_state.
        keep_going: If False (default), a model that raises stops the run. If True, the
            failure is recorded as an error result (flagged in model_metadata["error"],
            count 0) and the remaining models still run.
        evaluator: Fold evaluator; defaults to the Random Forest evaluator (TsT-RF).
        show_progress: Show tqdm progress bars.
        evaluator_factory: Builds one evaluator per model as ``factory(model, df_full, target_col)``
            (TsT-LLM, whose evaluator scores the model's zero-shot baseline when built).
            Overrides ``evaluator``.

    Returns:
        List of EvaluationResult objects, one per model evaluated.
    """
    if question_types is not None:
        selected = resolve_question_types(question_models, question_types) or []
        question_models = [m for m in question_models if m.name in selected]
    if not question_models:
        raise ValueError("No models provided for evaluation")

    cv_config = CrossValidationConfig(
        n_folds=n_splits,
        random_state=random_state,
        repeats=repeats,
        verbose=verbose,
        show_progress=show_progress,
        group_col=group_col,
        fold_seed=fold_seed,
    )
    cross_validator = UnifiedCrossValidator(cv_config)

    results: List[EvaluationResult] = []
    for model in question_models:
        logger.info(f"==== {model.name} ====")
        try:
            model_evaluator = evaluator
            if evaluator_factory is not None:
                resolved = UnifiedCrossValidator.resolve_target_column(model, target_col)
                model_evaluator = evaluator_factory(model, df_full, resolved)
            result = cross_validator.cross_validate(
                model=model, df=df_full, target_col=target_col, evaluator=model_evaluator
            )
        except Exception as e:
            if not keep_going:
                raise ModelEvaluationError(f"Evaluation failed for {model.name}: {e!r}") from e
            logger.error(f"Evaluation failed for {model.name}: {e!r}", exc_info=True)
            result = _create_error_result(model, repr(e))
        results.append(result)

    if verbose:
        logger.info(format_results_table(results))

    return results


# =============================================================================
# EVALUATION HELPER FUNCTIONS ------------------------------------------------
# =============================================================================


def _create_error_result(model: BiasModel, error_msg: str) -> EvaluationResult:
    """Placeholder result for a failed model (count 0, flagged in model_metadata["error"])."""
    error_fold = FoldResult(
        fold_id=1, score=0.0, train_size=0, test_idx=[], metric="acc", metadata={"error": error_msg}
    )
    error_repeat = RepeatResult.from_fold_results(0, [error_fold])

    return EvaluationResult.from_repeat_results(
        model_name=model.name,
        model_format=getattr(model, "format", "unknown"),
        metric_name=getattr(model, "metric", "unknown"),
        repeat_results=[error_repeat],
        model_metadata={"error": error_msg},
    )


def failed_results(results: List[EvaluationResult]) -> Dict[str, str]:
    """{model name: error message} for results produced by keep_going=True failures."""
    return {r.model_name: r.model_metadata["error"] for r in results if r.model_metadata.get("error")}


def summarize_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Overall statistics over the successfully evaluated models.

    Scores are fractions in [0, 1]; each model's score is accuracy (MC) or MRA (NUM).
    - weighted_mean: mean of the per-model scores weighted by question count
      (equivalently, the mean over all scored questions); the headline TsT-RF number.
    - macro_mean: unweighted mean of the per-model scores.
    - macro_std / macro_se / macro_t_95ci: spread of the per-repeat macro means
      (NaN with fewer than 2 repeats).
    - TsT-LLM results add zero_shot_weighted_mean / zero_shot_macro_mean (the base model's
      scores, same weighting) and delta_weighted / delta_macro (TsT minus zero-shot).
    Failed models (keep_going=True) are excluded and listed under "errors".
    """
    errors = failed_results(results)
    ok = [r for r in results if r.model_name not in errors]
    summary: Dict[str, Any] = {
        "n_models": len(ok),
        "total_count": int(sum(r.count for r in ok)),
        "weighted_mean": float("nan"),
        "weighted_std": float("nan"),
        "macro_mean": float("nan"),
        "repeats": None,
        "macro_std": float("nan"),
        "macro_se": float("nan"),
        "macro_t_95ci": [float("nan"), float("nan")],
        "errors": errors,
        "complete": not errors,
    }
    if not ok:
        return summary

    num_repeats = ok[0].repeats
    if any(r.repeats != num_repeats for r in ok):
        raise ValueError(f"All results must have the same number of repeats, got {[r.repeats for r in ok]}")

    scores = np.array([r.overall_mean for r in ok])
    counts = np.array([r.count for r in ok])
    weighted_avg, weighted_std = weighted_mean_std(scores, counts)

    repeat_macro = np.array([r.repeat_scores for r in ok]).mean(axis=0)  # (repeats,)
    summary.update(
        weighted_mean=float(weighted_avg),
        weighted_std=float(weighted_std),
        macro_mean=float(scores.mean()),
        repeats=int(num_repeats),
    )
    if all(r.zero_shot_sample_scores is not None for r in ok):
        zs = np.array([r.zero_shot_baseline for r in ok])
        zs_weighted = float((zs * counts).sum() / counts.sum())
        summary.update(
            zero_shot_weighted_mean=zs_weighted,
            zero_shot_macro_mean=float(zs.mean()),
            delta_weighted=float(weighted_avg) - zs_weighted,
            delta_macro=float(scores.mean()) - float(zs.mean()),
        )
    if num_repeats >= 2:
        from scipy.stats import t

        std = repeat_macro.std(ddof=1)
        se = std / np.sqrt(num_repeats)
        lo, hi = t.interval(0.95, num_repeats - 1, loc=repeat_macro.mean(), scale=se)
        summary.update(macro_std=float(std), macro_se=float(se), macro_t_95ci=[float(lo), float(hi)])
    return summary


def format_results_table(results: List[EvaluationResult]) -> str:
    """Human-readable per-model table plus the weighted and macro means (percent)."""
    errors = failed_results(results)
    llm = any(r.zero_shot_sample_scores is not None for r in results)
    rows = []
    for r in results:
        failed = r.model_name in errors
        row = {"question type": r.model_name, "format": r.model_format, "metric": r.metric_name}
        if llm:
            row["zero-shot"] = "" if failed else f"{100 * r.zero_shot_baseline:.2f}"
        row["score"] = "FAILED" if failed else f"{100 * r.overall_mean:.2f}"
        row["std"] = "" if failed else f"{100 * r.overall_std:.2f}"
        row["n"] = r.count
        rows.append(row)
    summary = summarize_results(results)
    lines = [pd.DataFrame(rows).to_string(index=False), ""]
    lines.append(
        f"weighted mean: {100 * summary['weighted_mean']:.2f}   macro mean: {100 * summary['macro_mean']:.2f}"
        f"   (questions scored: {summary['total_count']})"
    )
    if "zero_shot_weighted_mean" in summary:
        lines.append(
            f"zero-shot weighted mean: {100 * summary['zero_shot_weighted_mean']:.2f}   "
            f"delta (TsT - zero-shot), weighted: {100 * summary['delta_weighted']:+.2f}"
        )
    if summary["repeats"] and summary["repeats"] >= 2:
        lo, hi = summary["macro_t_95ci"]
        lines.append(f"macro over {summary['repeats']} repeats: 95% t-CI ({100 * lo:.2f}, {100 * hi:.2f})")
    if errors:
        lines.append(f"FAILED (excluded from the means): {', '.join(errors)}")
    return "\n".join(lines)


def _n_options(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return float(len(value))
    except TypeError:
        return np.nan


def per_question_scores(
    df: pd.DataFrame,
    models: List[BiasModel],
    results: List[EvaluationResult],
    id_col: str = "id",
) -> pd.DataFrame:
    """Per-question TsT score s(x), averaged over repeats.

    s(x) is the held-out prediction score of each question: the model's probability of
    the gold answer for multiple-choice (MC) questions, and the MRA of its prediction for
    numerical (NUM) questions. P(gold) is not comparable across formats or option counts,
    so the table adds ``n_options``, ``chance`` (1 / n_options for MC; empty for NUM) and
    ``s_minus_chance``. For TsT-LLM it also adds ``s_zero_shot`` (the base model's score)
    and ``delta`` (s minus s_zero_shot), the per-question gain that IBP ranks by. Rank
    questions within one format, never across formats.
    """
    errors = failed_results(results)
    frames = []
    for model, res in zip(models, results):
        if res.model_name in errors:
            continue
        per_idx: Dict[Any, List[float]] = {}
        for rep in res.repeat_results:
            for fold in rep.fold_results:
                for idx, score in (fold.sample_predictions or {}).items():
                    per_idx.setdefault(idx, []).append(score)
        if not per_idx:
            continue
        index = list(per_idx)
        rows = df.loc[index]
        out = pd.DataFrame(
            {
                id_col: rows[id_col].to_numpy(),
                "model": res.model_name,
                "question_type": rows["question_type"].to_numpy() if "question_type" in rows else res.model_name,
                "format": res.model_format,
                "metric": res.metric_name,
                "s": [float(np.mean(per_idx[i])) for i in index],
                "n_repeats": [len(per_idx[i]) for i in index],
            }
        )
        if res.model_format == "mc":
            if "n_options" in rows:
                n_opt = rows["n_options"].astype(float).to_numpy()
            else:
                src = "options" if "options" in rows else "choices"
                n_opt = rows[src].map(_n_options).to_numpy() if src in rows else np.full(len(rows), np.nan)
            out["n_options"] = pd.array(np.where(np.isnan(n_opt), np.nan, n_opt), dtype="Float64").astype("Int64")
            out["chance"] = 1.0 / out["n_options"].astype(float)
        else:
            out["n_options"] = pd.array([pd.NA] * len(out), dtype="Int64")
            out["chance"] = np.nan
        out["s_minus_chance"] = out["s"] - out["chance"]
        if res.zero_shot_sample_scores is not None:
            out["s_zero_shot"] = [float(res.zero_shot_sample_scores[i]) for i in index]
            out["delta"] = out["s"] - out["s_zero_shot"]
        frames.append(out)
    cols = [id_col, "model", "question_type", "format", "metric", "n_options", "chance", "s", "s_minus_chance"]
    if any("delta" in f for f in frames):
        cols += ["s_zero_shot", "delta"]
    if not frames:
        return pd.DataFrame(columns=cols + ["n_repeats"])
    return pd.concat(frames, ignore_index=True)[cols + ["n_repeats"]]


# =============================================================================
# BENCHMARK-LEVEL ENTRY POINT --------------------------------------------------
# =============================================================================


@dataclass
class BenchmarkRun:
    """A TsT-RF or TsT-LLM run on one benchmark: inputs, per-model results, and summaries."""

    benchmark: str
    hf_repo: Optional[str]
    revision: Optional[str]
    feature_set: str
    group_col: Optional[str]
    n_splits: int
    repeats: int
    random_state: int
    id_col: str
    target_col: Optional[str]
    dropped_features: List[str]
    df: pd.DataFrame = field(repr=False)
    models: List[BiasModel] = field(repr=False)
    results: List[EvaluationResult] = field(repr=False)
    fold_seed: Optional[int] = None
    mode: str = "rf"
    llm_config: Optional[Dict[str, Any]] = None

    @property
    def errors(self) -> Dict[str, str]:
        return failed_results(self.results)

    @property
    def weighted_mean(self) -> float:
        return summarize_results(self.results)["weighted_mean"]

    @property
    def macro_mean(self) -> float:
        return summarize_results(self.results)["macro_mean"]

    def s_x(self) -> pd.DataFrame:
        return per_question_scores(self.df, self.models, self.results, self.id_col)

    def summary(self) -> Dict[str, Any]:
        """Run settings, overall and per-question-type scores (strict JSON: NaN becomes None)."""
        from . import __version__

        stats = summarize_results(self.results)
        per_model = []
        for model, r in zip(self.models, self.results):
            failed = r.model_name in stats["errors"]
            per_model.append(
                {
                    "question_type": r.model_name,
                    "format": r.model_format,
                    "metric": r.metric_name,
                    "target_col": _quiet_target(model, self.target_col),
                    "count": int(r.count),
                    "score": None if failed else float(r.overall_mean),
                    "std": None if failed else float(r.overall_std),
                    "repeat_scores": None if failed else [float(x) for x in np.atleast_1d(r.repeat_scores)],
                    "fold_scores": None
                    if failed
                    else [[float(f.score) for f in rep.fold_results] for rep in r.repeat_results],
                    "feature_cols": list(getattr(model, "feature_cols", [])),
                    "error": stats["errors"].get(r.model_name),
                }
            )
            if self.mode == "llm":
                per_model[-1]["zero_shot"] = None if failed else float(r.zero_shot_baseline)
        return _json_safe(
            {
                "tst_version": __version__,
                "mode": self.mode,
                "benchmark": self.benchmark,
                "hf_repo": self.hf_repo,
                "revision": self.revision,
                "n_rows": int(len(self.df)),
                "id_col": self.id_col,
                "row_order_sha1": row_order_sha1(self.df, self.id_col) if self.id_col in self.df else None,
                "feature_set": self.feature_set if self.mode == "rf" else None,
                "dropped_features": self.dropped_features,
                "llm_config": self.llm_config,
                "folds": {
                    "n_splits": self.n_splits,
                    "group_col": self.group_col,
                    "repeats": self.repeats,
                    "random_state": self.random_state,
                    "fold_seed": self.fold_seed,
                },
                "score_scale": (
                    "fraction in [0, 1]; MC questions use accuracy, NUM questions use MRA"
                    if self.mode == "rf"
                    else "fraction in [0, 1]; MC questions use the probability of the gold option, NUM questions use MRA"
                ),
                **stats,
                "per_question_type": per_model,
                "environment": _environment(llm=self.mode == "llm"),
            }
        )

    def save(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Write summary.json and s_x.csv into ``output_dir``."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"summary": out / "summary.json", "s_x": out / "s_x.csv"}
        with open(paths["summary"], "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        self.s_x().to_csv(paths["s_x"], index=False, float_format="%.6g")
        return paths


def _quiet_target(model: BiasModel, target_col: Optional[str]) -> str:
    """resolve_target_column without re-emitting its warnings."""
    cv_logger = logging.getLogger(UnifiedCrossValidator.__module__)
    previous = cv_logger.disabled
    cv_logger.disabled = True
    try:
        return UnifiedCrossValidator.resolve_target_column(model, target_col)
    finally:
        cv_logger.disabled = previous


LLM_PACKAGES = ("torch", "vllm", "transformers", "peft", "trl", "accelerate", "datasets", "llamafactory")


def _environment(llm: bool = False) -> Dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    env: Dict[str, Any] = {"python": platform.python_version()}
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "huggingface-hub")
    for pkg in packages + (LLM_PACKAGES if llm else ()):
        try:
            env[pkg] = version(pkg)
        except PackageNotFoundError:
            env[pkg] = None
    # Packages whose versions differ from the lock (empty when the published values apply).
    env["differs_from_lock"] = version_mismatches()
    return env


def _json_safe(obj):
    """Replace NaN/inf with None so summary.json is strict JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def evaluate_benchmark(
    benchmark,
    *,
    mode: str = "rf",
    revision: Optional[str] = None,
    feature_set: str = "default",
    group_col: Optional[str] = None,
    n_splits: int = 5,
    repeats: int = 1,
    random_state: int = 42,
    fold_seed: Optional[int] = None,
    question_types: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    keep_going: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
    df: Optional[pd.DataFrame] = None,
    drop_features: Sequence[str] = (),
    llm_config=None,
    predictions_dir: Optional[Union[str, Path]] = None,
) -> BenchmarkRun:
    """Run TsT-RF (``mode="rf"``) or TsT-LLM (``mode="llm"``) on a registered benchmark.

    Args:
        benchmark: Benchmark name (e.g. "vsi", "cvb") or a Benchmark instance.
        mode: "rf" (Random Forest on hand-crafted features; CPU) or "llm" (LoRA fine-tuning
            of a text-only LLM; needs the ``llm`` extra and one GPU).
        revision: HF dataset revision; None uses the benchmark's pinned revision for ``mode``.
        feature_set: TsT-RF feature preset ("default" is leak-free; see the benchmark's feature_sets).
        llm_config: TsT-LLM settings (TsT.evaluators.llm.LLMRunConfig); None uses App. D.1.
        predictions_dir: TsT-LLM only: write per-question predictions (JSON lines) here.
        group_col: Keep rows sharing this column's value in one fold (e.g. "scene_name").
        df: Pre-loaded benchmark dataframe (skips load_data; ``revision`` is then only recorded).
        drop_features: Feature columns to remove from every model (ablations).
        Other arguments are passed to run_evaluation.
    """
    from .core.benchmark import BenchmarkRegistry

    bench = BenchmarkRegistry.get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    evaluator_factory = None
    if mode == "rf":
        models = bench.get_feature_based_models(feature_set=feature_set)
    elif mode == "llm":
        if feature_set != "default":
            raise ValueError("--feature_set applies to TsT-RF only")
        if drop_features:
            raise ValueError("drop_features applies to TsT-RF only")
        models = bench.get_qa_models()
        from .evaluators.llm import LLMEvaluator, LLMRunConfig

        llm_config = llm_config or LLMRunConfig()

        def evaluator_factory(model, df_full, target):
            return LLMEvaluator(
                model,
                df_full,
                target,
                llm_config=llm_config,
                seed=random_state,
                predictions_dir=predictions_dir,
                id_col=bench.id_col,
            )
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'rf' or 'llm'")
    # Validate the question types before any download.
    question_types = resolve_question_types(models, question_types)
    drop = sorted(set(drop_features))
    if drop:
        for m in models:
            m.feature_cols = [c for c in m.feature_cols if c not in drop]
    resolved_revision = bench.resolve_revision(revision, mode=mode)
    if df is None:
        df = bench.load_data(revision=resolved_revision)

    results = run_evaluation(
        question_models=models,
        df_full=df,
        n_splits=n_splits,
        random_state=random_state,
        verbose=verbose,
        repeats=repeats,
        question_types=question_types,
        target_col=target_col,
        group_col=group_col,
        fold_seed=fold_seed,
        keep_going=keep_going,
        show_progress=show_progress,
        evaluator_factory=evaluator_factory,
    )
    if question_types is not None:
        models = [m for m in models if m.name in question_types]

    return BenchmarkRun(
        benchmark=bench.name,
        hf_repo=bench.hf_repo,
        revision=resolved_revision,
        feature_set=feature_set,
        group_col=group_col,
        n_splits=n_splits,
        repeats=repeats,
        random_state=random_state,
        id_col=bench.id_col,
        target_col=target_col,
        dropped_features=drop,
        df=df,
        models=models,
        results=results,
        fold_seed=fold_seed,
        mode=mode,
        llm_config=llm_config.to_dict() if llm_config is not None else None,
    )


__all__ = [
    "BenchmarkRun",
    "ModelEvaluationError",
    "evaluate_benchmark",
    "failed_results",
    "format_results_table",
    "per_question_scores",
    "resolve_question_types",
    "row_order_sha1",
    "run_evaluation",
    "summarize_results",
    "version_mismatches",
]
