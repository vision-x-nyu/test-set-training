import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from .protocols import QType


# =============================================================================
# 1.  HELPERS ------------------------------------------------------------------
# =============================================================================


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """Label‑encode *object* columns (fit on **train only** to avoid leak).
    Unseen categories in test are mapped to -1."""
    cat_cols = X_train.select_dtypes(include="object").columns
    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        enc = LabelEncoder().fit(X_train[col].astype(str))
        mapping = {cls: i for i, cls in enumerate(enc.classes_)}
        X_train[col] = X_train[col].astype(str).map(mapping).astype(int)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)
        encoders[col] = enc
    return encoders


def mean_relative_accuracy(pred, true, start=0.5, end=0.95, step=0.05):
    thresholds = np.linspace(start, end, int((end - start) / step) + 2)
    rel_err = np.abs(pred - true) / true
    return np.mean([(rel_err < 1 - t).mean() for t in thresholds])


def _make_estimator(task, seed):
    if task == "clf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    else:
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def _score(est, X, y, metric="acc"):
    if metric == "acc":
        return est.score(X, y)  # plain accuracy
    elif metric == "mra":
        y_pred = est.predict(X)
        return mean_relative_accuracy(y_pred, y.values.astype(float))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def weighted_mean_std(scores: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """
    Weighted mean:
        weighted_avg = sum(score_i * count_i) / sum(count_i)
    Weighted variance:
        weighted_var = sum(count_i * (score_i - weighted_avg)**2) / sum(count_i)
    Weighted std:
        weighted_std = sqrt(weighted_var)
    Only models with count > 0 are included.
    """
    mask = counts > 0
    scores = scores[mask]
    counts = counts[mask]
    weighted_avg = (scores * counts).sum() / counts.sum() if counts.sum() > 0 else 0
    weighted_var = ((counts * (scores - weighted_avg) ** 2).sum() / counts.sum()) if counts.sum() > 0 else 0
    weighted_std = weighted_var**0.5
    return weighted_avg, weighted_std


# =============================================================================
# 4.  COMMON EVALUATION LOOP --------------------------------------------------
# =============================================================================


def evaluate_bias_model(
    model: QType,
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    repeats: int = 1,
    target_col: str = "ground_truth",
):
    qdf = model.select_rows(df)
    all_scores = []

    if model.target_col_override is not None and model.target_col_override != target_col:
        print(
            f"[WARNING] {model.name} has an override target column '{model.target_col_override}'. Replacing '{target_col}'."
        )
        target_col = model.target_col_override
    if model.task == "reg" and target_col == "gt_idx":
        # no gt_idx for regression tasks
        target_col = "ground_truth"
        print(f"[WARNING] {model.name} is numerical, with no gt_idx column. Overriding target column to 'ground_truth'")

    # Show progress bar over repeats
    repeat_pbar = tqdm(range(repeats), desc=f"[{model.name.upper()}] Repeats", disable=repeats == 1)

    for repeat in repeat_pbar:
        current_seed = random_state + repeat

        # Use appropriate splitter based on task type
        if model.task == "reg":
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf,)
        else:  # classification task
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf, qdf[target_col])

        scores: List[float] = []

        fold_pbar = tqdm(
            enumerate(splitter.split(*split_args), 1),
            desc=f"[{model.name.upper()}] Folds",
            total=n_splits,
            disable=repeats > 1,
        )
        for fold, (tr_idx, te_idx) in fold_pbar:
            tr, te = qdf.iloc[tr_idx].copy(), qdf.iloc[te_idx].copy()

            model.fit_feature_maps(tr)
            tr = model.add_features(tr)
            te = model.add_features(te)

            X_tr, X_te = tr[model.feature_cols].copy(), te[model.feature_cols].copy()
            encode_categoricals(X_tr, X_te)
            y_tr, y_te = tr[target_col], te[target_col]

            est = _make_estimator(model.task, current_seed)
            est.fit(X_tr, y_tr)
            scores.append(_score(est, X_te, y_te, model.metric))
            fold_pbar.set_postfix({f"fold_{model.metric}": f"{np.mean(scores):.2%}"})

        all_scores.append(scores)
        if repeats > 1:
            current_avg = np.mean(scores)
            repeat_pbar.set_postfix({f"avg_{model.metric}": f"{current_avg:.2%}"})

    # Calculate mean and std across all repeats
    mean_scores = [np.mean(scores) for scores in all_scores]
    mean_acc = float(np.mean(mean_scores))
    std_acc = float(np.std(mean_scores))
    count = len(qdf)

    if verbose:
        print(
            f"\n[{model.name.upper()}] Overall {model.metric.upper()}: {mean_acc:.2%} ± {std_acc:.2%} (n_splits={n_splits}, repeats={repeats})"
        )
        if repeats == 1:
            print(f"[{model.name.upper()}] Fold {model.metric.upper()}s: {[f'{s:.2%}' for s in all_scores[0]]}")
        else:
            print(f"[{model.name.upper()}] Repeat {model.metric.upper()}s: {[f'{s:.2%}' for s in mean_scores]}")

    # full‑data importances ---------------------------------------------------
    model.fit_feature_maps(qdf)  # all rows
    full_df = model.add_features(qdf.copy())
    X_full = full_df[model.feature_cols].copy()
    encode_categoricals(X_full, X_full.copy())
    y_full = full_df[target_col]

    est_full = _make_estimator(model.task, random_state)
    est_full.fit(X_full, y_full)
    fi = (
        pd.DataFrame({"feature": model.feature_cols, "importance": est_full.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    if verbose:
        print(f"\n[{model.name.upper()}] Feature importances:")
        print(fi.head(15))
    return mean_acc, std_acc, fi, count


def run_evaluation(
    models: List[QType],
    df_full: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False,
    repeats: int = 1,
    question_types: Union[List[str], None] = None,
    target_col: str = "ground_truth",
) -> pd.DataFrame:
    """
    Run evaluation for all models and return a summary table of results.

    Args:
        models: List of QType models to evaluate.
        df_full: The full dataframe containing all data.
        n_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed output during evaluation
        repeats: Number of times to repeat evaluation with different random seeds
        question_types: Optional list of question types to evaluate. If None, evaluate all types.
        target_col: Column to use as target variable (default: "ground_truth")

    Returns:
        DataFrame with model results including mean score and standard deviation
    """
    all_results = []

    # Filter models if question_types is specified
    if question_types is not None:
        models = [m for m in models if m.name in question_types]
        if not models:
            raise ValueError(f"Unknown question types: {question_types}")

    for m in models:
        print(f"\n================  {m.name.upper()}  ================")
        try:
            mean_score, std_score, fi, count = evaluate_bias_model(
                m,
                df_full,
                n_splits=n_splits,
                random_state=random_state,
                verbose=verbose,
                repeats=repeats,
                target_col=target_col,
            )
            weighted_score = mean_score * count
            all_results.append(
                {
                    "Model": m.name,
                    "Format": m.format.upper(),
                    "Metric": m.metric.upper(),
                    "Score": mean_score,
                    "± Std": std_score,
                    "Feature Importances": fi,
                    "Count": count,
                    "Weighted Score": weighted_score,
                    "Error": None,
                }
            )
        except Exception as e:
            print(f"[WARNING] Evaluation failed for model {m.name}: {e}")
            all_results.append(
                {
                    "Model": m.name,
                    "Format": getattr(m, "format", "N/A").upper() if hasattr(m, "format") else "N/A",
                    "Metric": getattr(m, "metric", "N/A").upper() if hasattr(m, "metric") else "N/A",
                    "Score": 0,
                    "± Std": 0,
                    "Feature Importances": None,
                    "Count": 0,
                    "Weighted Score": 0,
                    "Error": str(e),
                }
            )

    # Create summary table
    summary = pd.DataFrame(all_results)
    summary = summary.sort_values("Score", ascending=False)

    # Calculate overall average score
    overall_avg = summary["Score"].mean()
    overall_std = summary["Score"].std()
    total_count = summary["Count"].sum()

    # Weighted mean and std calculation (use raw values before formatting)
    weighted_avg, weighted_std = weighted_mean_std(summary["Score"].values, summary["Count"].values)

    # Format the scores as percentages (after all calculations)
    summary["Score"] = summary["Score"].map("{:.1%}".format)
    summary["± Std"] = summary["± Std"].map("{:.1%}".format)

    # Print pretty table
    print("\n" * 3 + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(summary[["Model", "Format", "Metric", "Score", "± Std", "Count"]].to_string(index=False))
    print("=" * 80)
    print(f"OVERALL AVERAGE SCORE: {overall_avg:.1%} ± {overall_std:.1%}")
    print(f"WEIGHTED AVERAGE SCORE: {weighted_avg:.1%} ± {weighted_std:.1%} (total examples: {total_count})")
    print("=" * 80)

    return summary
