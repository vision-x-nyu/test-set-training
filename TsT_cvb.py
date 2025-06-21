import os
import json
from functools import partial
from typing import Protocol, List, Tuple, Dict, Literal, Union
import re

import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, minmax_scale
from scipy.stats import lognorm


# =============================================================================
# 0.  DATA LOADING -------------------------------------------------------------
# =============================================================================

cvbench = load_dataset("nyu-visionx/CV-Bench")
df_full = cvbench["test"].to_pandas()
df_full["question_type"] = df_full["task"].str.lower() + "_" + df_full["type"].str.lower()
df_full["gt_idx"] = df_full["answer"].apply(lambda x: ord(x[1]) - ord('A'))
df_full["gt_option"] = df_full.apply(
    lambda row: row["choices"][row["gt_idx"]], axis=1
)
df_full["n_options"] = df_full["choices"].apply(len)



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
        X_test[col] = (
            X_test[col]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )
        encoders[col] = enc
    return encoders


# =============================================================================
# 2.  MICRO‑PROTOCOL FOR QUESTION TYPES ---------------------------------------
# =============================================================================

class QType(Protocol):
    name: str
    feature_cols: List[str]
    format: Literal["mc", "num"]

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect any statistics derived from *train* only (for leakage‑free CV)."""
        ...

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a **copy** of `df` with question‑specific feature columns added."""
        ...

    @property
    def task(self) -> Literal["clf", "reg"]:
        if self.format == "mc":
            return "clf"
        elif self.format == "num":
            return "reg"
        else:
            raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self) -> Literal["acc", "mra"]:
        return "acc"


# =============================================================================
# 3.  QUESTION TYPE IMPLEMENTATIONS -------------------------------------------
# =============================================================================


"""MC QUESTIONS"""


# 2D Count
class Count2DModel(QType):
    name = "count_2d"
    format = "mc"

    _choice_dist_cols = [
        f"choice_{i}_dist_from_obj_mean"
        for i in range(4)
    ]
    _choice_dist_from_global_cols = [
        f"choice_{i}_dist_from_global_mean"
        for i in range(4)
    ]

    feature_cols = [
        "n_options",
        "object",
        "obj_count",
        "obj_freq_score",
        "obj_val_mean",
        "obj_val_std",
        "obj_val_log_mean",
        "obj_val_log_std",
        "obj_val_log_ratio",
        "global_mean_log",
        "global_std_log",
        *_choice_dist_cols,
        *_choice_dist_from_global_cols,
        # # NOTE: the below features leverage privileged gt info. Remove?
        # "gt_global_mean_dist_score",
        # "gt_log_obj_mean_dist_score",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_stats: pd.DataFrame | None = None
        self.global_mean_log: float | None = None
        self.global_std_log: float | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess 2D object counting questions."""
        qdf = df[df["question_type"] == self.name].copy()
        
        # Extract object name from question
        qdf["object"] = qdf["question"].str.extract(
            r'How many (.*?) are in the image')[0].str.strip()
        
        # Use preprocessed ground truth
        qdf["ground_truth"] = pd.to_numeric(qdf["gt_option"], errors="coerce")
        
        # Add log-transformed ground truth
        qdf["log_ground_truth"] = np.log10(qdf["ground_truth"] + 1.0)
        
        # Drop rows where extraction failed
        qdf.dropna(subset=["object", "ground_truth"], inplace=True)
        
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object statistics and global stats from training data."""
        # Calculate object statistics
        self.obj_stats = train_df.groupby("object").agg(
            obj_count=("idx", "count"),
            obj_val_mean=("ground_truth", "mean"),
            obj_val_std=("ground_truth", "std"),
            obj_val_log_mean=("log_ground_truth", "mean"),
            obj_val_log_std=("log_ground_truth", "std")
        ).reset_index()

        # Handle std=0 cases
        epsilon = 1e-6
        self.obj_stats["obj_val_std"] = self.obj_stats["obj_val_std"].fillna(0)
        self.obj_stats["obj_val_log_std"] = self.obj_stats["obj_val_log_std"].fillna(0)

        # Calculate ratios
        self.obj_stats["obj_val_log_ratio"] = (
            self.obj_stats["obj_val_log_std"] / 
            (self.obj_stats["obj_val_log_mean"] + epsilon)
        ).fillna(0)

        # Calculate global statistics
        self.global_mean_log = train_df["log_ground_truth"].mean()
        self.global_std_log = train_df["log_ground_truth"].std()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and statistical features to the dataframe."""
        if self.obj_stats is None or self.global_mean_log is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        epsilon = 1e-6

        # Merge with object statistics
        df = pd.merge(df, self.obj_stats, on="object", how="left")

        # Calculate frequency score
        df["obj_freq_score"] = minmax_scale(df["obj_count"])

        # Calculate inverse variance score
        df["obj_val_log_ratio"] = 1.0 - minmax_scale(df["obj_val_log_ratio"] + epsilon)

        # Add global statistics
        df["global_mean_log"] = self.global_mean_log
        df["global_std_log"] = self.global_std_log

        # For each choice, calculate distance from typical
        for i, choice in enumerate(df["choices"]):
            choice_val = pd.to_numeric(choice, errors="coerce")
            choice_log = np.log10(choice_val + 1.0)
            
            # Get the object's mean for this row
            obj_mean = df.loc[df.index[i], "obj_val_log_mean"]
            obj_std = df.loc[df.index[i], "obj_val_log_std"]
            
            # Calculate distances
            df.loc[df.index[i], f"choice_{i}_dist_from_obj_mean"] = abs(choice_log - obj_mean)
            df.loc[df.index[i], f"choice_{i}_dist_from_global_mean"] = abs(choice_log - self.global_mean_log)

        #  Privileged features
        # ---------------------------------------
        # Calculate distance from object mean score
        gt_norm_dist = abs(df["log_ground_truth"] - df["obj_val_log_mean"]) / (
            df["obj_val_log_std"] + epsilon
        )
        df["gt_log_obj_mean_dist_score"] = 1.0 - minmax_scale(gt_norm_dist + epsilon)

        # Calculate global distance score
        gt_global_dist = abs(df["log_ground_truth"] - self.global_mean_log) / (
            self.global_std_log + epsilon
        )
        df["gt_global_mean_dist_score"] = 1.0 - minmax_scale(gt_global_dist + epsilon)


        return df


# 2D Relation
class Relation2DModel(QType):
    name = "relation_2d"
    format = "mc"

    feature_cols = [
        "n_options",
        "object_1",
        "object_2",
        "pair_freq_score",
        "pair_answer_freq_score",
        "contains_left",         # NEW: Question contains "left"
        "contains_right",        # NEW: Question contains "right"
        "contains_above",        # NEW: Question contains "above"
        "contains_below",        # NEW: Question contains "below"
        "contains_front",        # NEW: Question contains "front"
        "contains_behind",       # NEW: Question contains "behind"
        "spatial_keyword_count", # NEW: Total spatial keywords
        "question_length",       # NEW: Question length
        "is_majority_answer",    # NEW: Is this the most common answer?
        # "answer_position_bias",  # NEW: How often this position is correct
        # "answer_entropy",        # NEW: Entropy of answer distribution
    ]

    def __init__(self):
        self.pair_freq_map: pd.Series | None = None
        self.pair_answer_freq_map: Dict[Tuple[str, str], Dict[str, float]] | None = None
        self.answer_position_freq: Dict[int, float] | None = None  # NEW
        self.majority_answer: str | None = None  # NEW
        self.answer_distribution: pd.Series | None = None  # NEW

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess 2D relation questions."""
        qdf = df[df["question_type"] == self.name].copy()
        
        # Extract object pairs from question
        def extract_objects(question):
            # Remove annotation markers
            question = question.replace(' (annotated by the red box)', '')
            question = question.replace(' (annotated by the blue box)', '')
            
            # Try to find patterns like "the X and the Y"
            match = re.search(r'the relative positions of the ([^,]+?) and the ([^,]+?)[, ]', question)
            if match:
                return match.group(1).strip(), match.group(2).strip()
            
            # Fallback: try to find "the X" and "the Y" separately
            matches = re.findall(r'the ([a-zA-Z0-9_ ]+?)[,?\.]', question)
            if len(matches) >= 2:
                return matches[0].strip(), matches[1].strip()
            return None, None

        # Extract object pairs and sort them for consistency
        qdf[["object_1", "object_2"]] = qdf["question"].apply(
            lambda q: pd.Series(sorted(extract_objects(q)))
        )
        
        # Drop rows where extraction failed
        qdf.dropna(subset=["object_1", "object_2"], inplace=True)
        
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies and answer distributions from training data."""
        # Calculate pair frequencies
        pairs = train_df.apply(
            lambda row: f"{row['object_1']}-{row['object_2']}", 
            axis=1
        )
        self.pair_freq_map = pairs.value_counts(normalize=True)
        
        # Calculate answer frequencies for each pair
        self.pair_answer_freq_map = {}
        for _, row in train_df.iterrows():
            pair = (row["object_1"], row["object_2"])
            answer = row["gt_option"]
            
            if pair not in self.pair_answer_freq_map:
                self.pair_answer_freq_map[pair] = {}
            
            if answer not in self.pair_answer_freq_map[pair]:
                self.pair_answer_freq_map[pair][answer] = 0
            self.pair_answer_freq_map[pair][answer] += 1
        
        # Normalize answer frequencies for each pair
        for pair in self.pair_answer_freq_map:
            total = sum(self.pair_answer_freq_map[pair].values())
            self.pair_answer_freq_map[pair] = {
                ans: count/total 
                for ans, count in self.pair_answer_freq_map[pair].items()
            }

        # NEW: Calculate answer position frequencies
        self.answer_position_freq = train_df["gt_idx"].value_counts(normalize=True).to_dict()

        # NEW: Find majority answer
        self.answer_distribution = train_df["gt_option"].value_counts()
        self.majority_answer = self.answer_distribution.idxmax()

        # # Print diagnostic info
        # print(f"\n[DIAGNOSTIC] Answer distribution in training:")
        # print(self.answer_distribution.sort_index())
        # print(f"Majority answer: {self.majority_answer} ({self.answer_distribution[self.majority_answer]/len(train_df)*100:.1f}%)")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and spatial features to the dataframe."""
        if self.pair_freq_map is None or self.pair_answer_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        
        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1
        )
        
        # Calculate answer frequency score for the pair
        df["pair_answer_freq_score"] = df.apply(
            lambda row: self.pair_answer_freq_map.get(
                (row["object_1"], row["object_2"]), {}
            ).get(row["gt_option"], 0),
            axis=1
        )
        
        # NEW: Answer position bias
        df["answer_position_bias"] = df["gt_idx"].map(self.answer_position_freq).fillna(0)
        
        # NEW: Spatial keyword features
        spatial_keywords = {
            "left": ["left", "to the left"],
            "right": ["right", "to the right"],
            "above": ["above", "over", "on top"],
            "below": ["below", "under", "beneath"],
            "front": ["front", "in front", "foreground"],
            "behind": ["behind", "back", "background"]
        }
        
        for direction, keywords in spatial_keywords.items():
            df[f"contains_{direction}"] = df["question"].apply(
                lambda q: int(any(kw in q.lower() for kw in keywords))
            )
        
        # NEW: Total spatial keywords
        df["spatial_keyword_count"] = sum(df[f"contains_{direction}"] for direction in spatial_keywords)
        
        # NEW: Question length
        df["question_length"] = df["question"].str.len()
        
        # NEW: Is majority answer
        df["is_majority_answer"] = (df["gt_option"] == self.majority_answer).astype(int)
        
        # NEW: Answer entropy (diversity of answers)
        total_answers = sum(self.answer_distribution.values)
        probs = [count/total_answers for count in self.answer_distribution.values]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        df["answer_entropy"] = entropy
        
        return df


# 3D Depth
class Depth3DModel(QType):
    name = "depth_3d"
    format = "mc"

    feature_cols = [
        "object_1",
        "object_2",
        "pair_freq_score",
        "pair_answer_freq_score",
        "n_options",  # Number of choices available
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.pair_freq_map: pd.Series | None = None
        self.pair_answer_freq_map: Dict[Tuple[str, str], Dict[str, float]] | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess 3D depth questions."""
        qdf = df[df["question_type"] == self.name].copy()
        
        # For depth questions, the choices themselves are the objects being compared
        # Sort the choices to ensure consistent pairing
        qdf[["object_1", "object_2"]] = qdf["choices"].apply(
            lambda x: pd.Series(sorted(x))
        )
        
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies and answer distributions from training data."""
        # Calculate pair frequencies
        pairs = train_df.apply(
            lambda row: f"{row['object_1']}-{row['object_2']}", 
            axis=1
        )
        self.pair_freq_map = pairs.value_counts(normalize=True)
        
        # Calculate answer frequencies for each pair
        self.pair_answer_freq_map = {}
        for _, row in train_df.iterrows():
            pair = (row["object_1"], row["object_2"])
            answer = row["gt_option"]
            
            if pair not in self.pair_answer_freq_map:
                self.pair_answer_freq_map[pair] = {}
            
            if answer not in self.pair_answer_freq_map[pair]:
                self.pair_answer_freq_map[pair][answer] = 0
            self.pair_answer_freq_map[pair][answer] += 1
        
        # Normalize answer frequencies for each pair
        for pair in self.pair_answer_freq_map:
            total = sum(self.pair_answer_freq_map[pair].values())
            self.pair_answer_freq_map[pair] = {
                ans: count/total 
                for ans, count in self.pair_answer_freq_map[pair].items()
            }

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.pair_freq_map is None or self.pair_answer_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        
        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1
        )
        
        # Calculate answer frequency score for the pair
        df["pair_answer_freq_score"] = df.apply(
            lambda row: self.pair_answer_freq_map.get(
                (row["object_1"], row["object_2"]), {}
            ).get(row["gt_option"], 0),
            axis=1
        )
        
        return df


# 3D Distance
class Distance3DModel(QType):
    name = "distance_3d"
    format = "mc"

    feature_cols = [
        "object_1",
        "object_2",
        "pair_freq_score",
        "pair_answer_freq_score",
        "n_options",  # Number of choices available
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.pair_freq_map: pd.Series | None = None
        self.pair_answer_freq_map: Dict[Tuple[str, str], Dict[str, float]] | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess 3D distance questions."""
        qdf = df[df["question_type"] == self.name].copy()
        
        # For distance questions, the choices themselves are the objects being compared
        # Sort the choices to ensure consistent pairing
        qdf[["object_1", "object_2"]] = qdf["choices"].apply(
            lambda x: pd.Series(sorted(x))
        )
        
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies and answer distributions from training data."""
        # Calculate pair frequencies
        pairs = train_df.apply(
            lambda row: f"{row['object_1']}-{row['object_2']}", 
            axis=1
        )
        self.pair_freq_map = pairs.value_counts(normalize=True)
        
        # Calculate answer frequencies for each pair
        self.pair_answer_freq_map = {}
        for _, row in train_df.iterrows():
            pair = (row["object_1"], row["object_2"])
            answer = row["gt_option"]
            
            if pair not in self.pair_answer_freq_map:
                self.pair_answer_freq_map[pair] = {}
            
            if answer not in self.pair_answer_freq_map[pair]:
                self.pair_answer_freq_map[pair][answer] = 0
            self.pair_answer_freq_map[pair][answer] += 1
        
        # Normalize answer frequencies for each pair
        for pair in self.pair_answer_freq_map:
            total = sum(self.pair_answer_freq_map[pair].values())
            self.pair_answer_freq_map[pair] = {
                ans: count/total 
                for ans, count in self.pair_answer_freq_map[pair].items()
            }

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.pair_freq_map is None or self.pair_answer_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        
        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1
        )
        
        # Calculate answer frequency score for the pair
        df["pair_answer_freq_score"] = df.apply(
            lambda row: self.pair_answer_freq_map.get(
                (row["object_1"], row["object_2"]), {}
            ).get(row["gt_option"], 0),
            axis=1
        )
        
        return df


# =============================================================================
# 4.  COMMON EVALUATION LOOP --------------------------------------------------
# =============================================================================

# utils --------------------------------------------------------------------
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


def evaluate_bias_model(
    model: QType,
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    repeats: int = 1,
    target_col: str = "gt_idx",
):
    qdf = model.select_rows(df)
    all_scores = []

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

        fold_pbar = tqdm(enumerate(splitter.split(*split_args), 1), desc=f"[{model.name.upper()}] Folds", total=n_splits, disable=repeats > 1)
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

    if verbose:
        print(f"\n[{model.name.upper()}] Overall {model.metric.upper()}: {mean_acc:.2%} ± {std_acc:.2%} (n_splits={n_splits}, repeats={repeats})")
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
    return mean_acc, std_acc, fi


# =============================================================================
# 6.  MAIN --------------------------------------------------------------------
# =============================================================================

def run_evaluation(n_splits: int = 5, random_state: int = 42, verbose: bool = False, repeats: int = 1, question_types: Union[List[str], None] = None, target_col: str = "gt_idx") -> pd.DataFrame:
    """
    Run evaluation for all models and return a summary table of results.

    Args:
        n_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed output during evaluation
        repeats: Number of times to repeat evaluation with different random seeds
        question_types: Optional list of question types to evaluate. If None, evaluate all types.
        target_col: Column to use as target variable (default: "gt_idx")

    Returns:
        DataFrame with model results including mean score and standard deviation
    """
    all_results = []

    # Create models list once
    models = [
        ## MC
        Count2DModel(),
        Relation2DModel(),
        Depth3DModel(),
        Distance3DModel(),  # Uncommented to include in evaluation
    ]

    # Filter models if question_types is specified
    if question_types is not None:
        models = [m for m in models if m.name in question_types]
        if not models:
            raise ValueError(f"Unknown question types: {question_types}")

    for m in models:
        print(f"\n================  {m.name.upper()}  ================")
        mean_score, std_score, fi = evaluate_bias_model(
            m,
            df_full,
            n_splits=n_splits,
            random_state=random_state,
            verbose=verbose,
            repeats=repeats,
            target_col=target_col,
        )
        all_results.append({
            "Model": m.name,
            "Format": m.format.upper(),
            "Metric": m.metric.upper(),
            "Score": mean_score,
            "± Std": std_score,
            "Feature Importances": fi,
        })

    # Create summary table
    summary = pd.DataFrame(all_results)
    summary = summary.sort_values("Score", ascending=False)

    # Calculate overall average score
    overall_avg = summary["Score"].mean()
    overall_std = summary["Score"].std()

    # Format the scores as percentages
    summary["Score"] = summary["Score"].map("{:.1%}".format)
    summary["± Std"] = summary["± Std"].map("{:.1%}".format)

    # Print pretty table
    print("\n"*3 + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(summary[["Model", "Format", "Metric", "Score", "± Std"]].to_string(index=False))
    print("="*80)
    print(f"OVERALL AVERAGE SCORE: {overall_avg:.1%} ± {overall_std:.1%}")
    print("="*80)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", "-k", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--random_state", "-s", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed output"
    )
    parser.add_argument(
        "--repeats", "-r", type=int, default=1, 
        help="Number of times to repeat evaluation with different random seeds"
    )
    parser.add_argument(
        "--question_types", "-q", type=str, default=None,
        help="Comma-separated list of question types to evaluate (e.g. 'Count,Relation')"
    )
    parser.add_argument(
        "--target_col", "-t", type=str, default="gt_idx",
        help="Column to use as target variable (default: gt_idx)"
    )
    args = parser.parse_args()

    # Parse question types if provided
    question_types = None
    if args.question_types is not None:
        question_types = [q.strip() for q in args.question_types.split(",")]

    run_evaluation(
        n_splits=args.n_splits,
        random_state=args.random_state,
        verbose=args.verbose,
        repeats=args.repeats,
        question_types=question_types,
        target_col=args.target_col
    )
