"""
Iterative Bias Pruning (IBP).

Each iteration runs TsT on the remaining questions, ranks them by their held-out score, and
removes the top batch (subject to the strategy's group floor), until the budget is spent or
the highest remaining score drops below an early-stop threshold.

- :func:`iterative_bias_pruning`: the loop on one DataFrame, given a callback that runs TsT.
- :func:`debias_benchmark`: a registered benchmark end to end (load at a pinned revision, split
  the budget into groups, run TsT-RF or TsT-LLM through :func:`TsT.evaluation.evaluate_benchmark`).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..core.protocols import EvaluationResult, IBPStrategy
from .allocation import Alloc, plan_groups
from .config import IBPConfig, Score
from .results import GroupResult, IBPIterationResult, IBPResult, IBPRun
from .strategies import TIE_BREAKS, MissingScoreError

logger = logging.getLogger(__name__)

RunTsT = Callable[[pd.DataFrame], List[EvaluationResult]]


def aggregate_sample_predictions(results: Sequence[EvaluationResult], score: Score = "s") -> Dict[Any, float]:
    """Per-question bias score from TsT results, keyed by DataFrame index label.

    ``score="s"``: the held-out score s(x), averaged over repeats (and over models, if several
    score the same question). ``score="delta"``: s(x) minus the question's zero-shot score
    (``EvaluationResult.zero_shot_sample_scores``, set by TsT-LLM).

    Raises:
        ValueError: a result failed, has no per-question predictions, or (delta) has no
            zero-shot scores.
        MissingScoreError: (delta) a scored question has no zero-shot score.
    """
    values: defaultdict[Any, List[float]] = defaultdict(list)
    for res in results:
        if res.model_metadata.get("error"):
            raise ValueError(f"TsT failed for {res.model_name}: {res.model_metadata['error']}")
        zero_shot = res.zero_shot_sample_scores
        if score == "delta" and zero_shot is None:
            raise ValueError(
                f"score='delta' needs per-question zero-shot scores, which {res.model_name} does not have "
                "(only TsT-LLM results do; rank TsT-RF results by score='s')"
            )
        for rep in res.repeat_results:
            for fold in rep.fold_results:
                if fold.sample_predictions is None:
                    raise ValueError(f"{res.model_name} fold {fold.fold_id} has no per-question predictions")
                for idx, s in fold.sample_predictions.items():
                    if score == "delta":
                        if idx not in zero_shot:
                            raise MissingScoreError(f"{res.model_name}: question {idx!r} has no zero-shot score")
                        s = s - zero_shot[idx]
                    values[idx].append(float(s))
    return {idx: sum(v) / len(v) for idx, v in values.items()}


def iterative_bias_pruning(
    df: pd.DataFrame,
    strategy: IBPStrategy,
    config: IBPConfig,
    run_tst_fn: RunTsT,
    ranked_index_fn: Optional[Callable[[pd.DataFrame], pd.Index]] = None,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, IBPResult]:
    """Run IBP on one DataFrame.

    Args:
        df: The questions to prune (not modified).
        strategy: Scores and selects candidates (e.g. a TopKDiversityStrategy).
        config: Budget, batch size, early stop and score type.
        run_tst_fn: Runs TsT on the current questions and returns its results.
        ranked_index_fn: Index labels of the questions to rank (those some model scores);
            None ranks every row. Rows outside it are never removed but count toward group sizes.
            Every ranked question must get a score, or MissingScoreError is raised.
        id_col: Column with the question ids recorded in the result (index labels if None).

    Returns:
        (kept DataFrame, IBPResult)
    """
    df = df.copy()
    original_size = len(df)
    removed: List[Any] = []
    removed_ids: List[Any] = []
    iterations: List[IBPIterationResult] = []
    stop_reason = "budget"

    def ids_of(labels: List[Any]) -> List[Any]:
        if id_col is not None and id_col in df.columns:
            return [_plain(x) for x in df.loc[labels, id_col].tolist()]
        return [_plain(x) for x in labels]

    while len(removed) < config.budget:
        iteration = len(iterations) + 1
        sample_predictions = aggregate_sample_predictions(run_tst_fn(df), config.score)
        ranked = df.index if ranked_index_fn is None else df.index[df.index.isin(ranked_index_fn(df))]
        bias_scores = strategy.compute_bias_scores(df.loc[ranked], sample_predictions)
        max_bias = float(bias_scores.max()) if len(bias_scores) else float("nan")
        mean_bias = float(bias_scores.mean()) if len(bias_scores) else float("nan")

        # Algorithm 1: stop when max_i s_i <= tau (or when nothing is left to rank)
        if config.early_stop_threshold is not None and not max_bias > config.early_stop_threshold:
            stop_reason = "early_stop"
            logger.info(f"IBP early stop: max bias {max_bias:.4f} <= {config.early_stop_threshold}")
            break

        batch = min(config.batch_size, config.budget - len(removed))
        candidates = list(strategy.select_removal_candidates(df, bias_scores, batch))
        if not candidates:
            stop_reason = "no_candidates"
            logger.warning("IBP: no question can be removed without breaking the group floor; stopping")
            break

        cand_ids = ids_of(candidates)
        iterations.append(
            IBPIterationResult(
                iteration=iteration,
                removed_indices=[_plain(x) for x in candidates],
                removed_ids=cand_ids,
                max_bias_score=max_bias,
                mean_bias_score=mean_bias,
                n_ranked=len(bias_scores),
                dataset_size_before=len(df),
                dataset_size_after=len(df) - len(candidates),
            )
        )
        log = logger.info if config.verbose else logger.debug
        log(
            f"IBP iteration {iteration}: {len(df)} questions, max bias {max_bias:.4f}, mean {mean_bias:.4f}; removing {len(candidates)}"
        )
        df = df.drop(index=candidates)
        removed.extend(candidates)
        removed_ids.extend(cand_ids)

    result = IBPResult(
        removed_indices=[_plain(x) for x in removed],
        removed_ids=removed_ids,
        iterations=iterations,
        config=config,
        original_size=original_size,
        final_size=len(df),
        stop_reason=stop_reason,
    )
    return df, result


def _plain(x: Any) -> Any:
    """numpy scalar -> Python scalar (for JSON)."""
    return x.item() if hasattr(x, "item") else x


def _versions(mode: str = "rf") -> Dict[str, Any]:
    from .. import __version__
    from ..evaluation import _environment

    return {"tst_version": __version__, **_environment(llm=mode == "llm")}


def read_id_list(path: Union[str, Path]) -> List[str]:
    """Ids from a text file, one per line (blank lines ignored)."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def debias_benchmark(
    benchmark,
    *,
    mode: str = "rf",
    alloc: Alloc = "per_format",
    budget: Optional[int] = None,
    frac: Optional[float] = None,
    budgets: Optional[Mapping[str, int]] = None,
    budgets_source: Optional[str] = None,
    batch_size: int = 50,
    score: Optional[Score] = None,
    tie_break: str = "id",
    min_per_group: Optional[int] = None,
    early_stop_threshold: Optional[float] = None,
    revision: Optional[str] = None,
    feature_set: str = "default",
    n_splits: int = 5,
    repeats: int = 1,
    random_state: int = 42,
    fold_seed: Optional[int] = None,
    group_col: Optional[str] = None,
    target_col: Optional[str] = None,
    llm_config=None,
    df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
    on_group_done: Optional[Callable[[GroupResult], None]] = None,
) -> IBPRun:
    """Run IBP on a registered benchmark.

    Args:
        benchmark: Benchmark name or instance (must define ``get_ibp_strategy``).
        mode: "rf" (TsT-RF; CPU) or "llm" (TsT-LLM; one GPU, ``llm`` extra).
        alloc, budget, frac, budgets: How the budget is split (see TsT.debiasing.allocation).
        budgets_source: Label recorded in the summary for where ``budgets`` came from.
        batch_size: Questions removed per iteration (capped at each group's budget).
        score: "s" (raw held-out score) or "delta" (s minus zero-shot; TsT-LLM only).
            Default: "s" for TsT-RF, "delta" for TsT-LLM.
        tie_break: "id" (default; stable, by question id) or "legacy" (proceedings' unstable sort).
        min_per_group: Override the strategy's group floor.
        early_stop_threshold: Stop a group when its highest bias score is at or below this value.
        revision: Dataset revision (None: the benchmark's pinned revision for ``mode``).
        df: Pre-loaded benchmark DataFrame (skips load_data; ``revision`` is then only recorded).
        Other arguments are passed to :func:`TsT.evaluation.evaluate_benchmark` on every iteration.
        on_group_done: Called with each finished group (progress reporting).
    """
    from ..core.benchmark import BenchmarkRegistry
    from ..evaluation import evaluate_benchmark
    from ..utils import row_order_sha1

    bench = BenchmarkRegistry.get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
    if mode not in ("rf", "llm"):
        raise ValueError(f"Unknown mode {mode!r}; use 'rf' or 'llm'")
    score = score or ("s" if mode == "rf" else "delta")
    if mode == "rf" and score == "delta":
        raise ValueError("score='delta' needs zero-shot scores, which only TsT-LLM has; use score='s' with --mode rf")
    if mode == "llm" and feature_set != "default":
        raise ValueError("feature_set applies to TsT-RF only")
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}, got {tie_break!r}")
    if mode == "llm" and llm_config is None:
        from ..evaluators.llm import LLMRunConfig

        llm_config = LLMRunConfig()

    strategy = bench.get_ibp_strategy()
    if hasattr(strategy, "tie_break"):
        strategy.tie_break = tie_break
    if getattr(strategy, "id_col", "unset") is None:
        strategy.id_col = bench.id_col
    if min_per_group is not None:
        strategy.min_samples_per_group = min_per_group
    floor = int(getattr(strategy, "min_samples_per_group", 0))

    models = bench.get_feature_based_models(feature_set=feature_set) if mode == "rf" else bench.get_qa_models()
    resolved_revision = bench.resolve_revision(revision, mode=mode)
    if df is None:
        df = bench.load_data(revision=resolved_revision)
    id_col = bench.id_col
    if id_col not in df.columns:
        raise ValueError(f"The benchmark data have no id column {id_col!r}")
    if df[id_col].duplicated().any():
        raise ValueError(f"Question ids in {id_col!r} are not unique")

    # Rows each model scores (select_rows is a row filter, so this is computed once on all rows).
    coverage = {m.name: m.select_rows(df).index for m in models}
    scorable = pd.Index([])
    for index in coverage.values():
        scorable = scorable.union(index)
    # Per-format and per-type budgets are split over the questions some model scores (e.g. not MMMU's
    # open-ended questions under TsT-LLM); a global run keeps unscored questions toward the floors.
    df_alloc = df if alloc == "global" else df.loc[df.index.isin(scorable)]
    groups = plan_groups(df_alloc, alloc, budget=budget, frac=frac, budgets=budgets, min_per_group=floor)
    for group in groups:
        if not any(coverage[m.name].isin(group.index).any() for m in models):
            raise ValueError(f"No {mode} model scores any question in group {group.name!r}")

    def run_tst_for(model_names: List[str]) -> RunTsT:
        def run(d: pd.DataFrame) -> List[EvaluationResult]:
            return evaluate_benchmark(
                bench,
                mode=mode,
                revision=resolved_revision,
                feature_set=feature_set,
                group_col=group_col,
                n_splits=n_splits,
                repeats=repeats,
                random_state=random_state,
                fold_seed=fold_seed,
                question_types=model_names,
                target_col=target_col,
                keep_going=False,
                verbose=False,
                show_progress=False,
                df=d,
                llm_config=llm_config,
            ).results

        return run

    group_results: List[GroupResult] = []
    removed_labels: List[Any] = []
    removed_ids: List[Any] = []
    for group in groups:
        df_g = df.loc[group.index]
        names = [m.name for m in models if coverage[m.name].isin(group.index).any()]
        scored = pd.Index([])
        for name in names:
            scored = scored.union(coverage[name])

        def ranked_index(d: pd.DataFrame, _scored=scored) -> pd.Index:
            return d.index[d.index.isin(_scored)]

        config = IBPConfig(
            budget=group.budget,
            batch_size=min(batch_size, group.budget),
            early_stop_threshold=early_stop_threshold,
            score=score,
            verbose=verbose,
        )
        _, result = iterative_bias_pruning(
            df_g, strategy, config, run_tst_for(names), ranked_index_fn=ranked_index, id_col=id_col
        )
        gr = GroupResult(
            name=group.name,
            size=group.size,
            n_ranked=len(ranked_index(df_g)),
            budget=group.budget,
            models=names,
            result=result,
        )
        group_results.append(gr)
        removed_labels.extend(result.removed_indices)
        removed_ids.extend(result.removed_ids)
        if on_group_done is not None:
            on_group_done(gr)

    kept_ids = [_plain(x) for x in df.drop(index=removed_labels)[id_col].tolist()]
    settings = {
        "benchmark": bench.name,
        "hf_repo": bench.hf_repo,
        "revision": resolved_revision,
        "id_col": id_col,
        "row_order_sha1": row_order_sha1(df, id_col),
        "mode": mode,
        "alloc": alloc,
        "budget": budget,
        "frac": frac,
        "budgets": {g.name: g.budget for g in groups},
        "budgets_source": budgets_source,
        "batch_size": batch_size,
        "score": score,
        "tie_break": tie_break,
        "min_per_group": floor,
        "group_column": getattr(strategy, "group_column", None),
        "early_stop_threshold": early_stop_threshold,
        "feature_set": feature_set if mode == "rf" else None,
        "llm_config": llm_config.to_dict() if llm_config is not None and hasattr(llm_config, "to_dict") else None,
        "n_splits": n_splits,
        "repeats": repeats,
        "random_state": random_state,
        "fold_seed": fold_seed,
        "group_col": group_col,
        "target_col": target_col,
    }
    return IBPRun(
        settings=settings,
        groups=group_results,
        removed_ids=removed_ids,
        kept_ids=kept_ids,
        n_rows=len(df),
        environment=_versions(mode),
    )
