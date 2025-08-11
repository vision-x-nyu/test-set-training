from typing import Tuple, Dict
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from ...core.protocols import FeatureBasedBiasModel
from ezcolorlog import root_logger as logger


# =============================================================================
# GLOBAL MODEL ---------------------------------------------------------------
# =============================================================================


# TODO: not finished, clean this up
class CVBModel(FeatureBasedBiasModel):
    name = "cvb"
    format = "mc"

    feature_cols = []

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


# =============================================================================
# MULTIPLE CHOICE QUESTION MODELS ---------------------------------------------
# =============================================================================


# 2D Count
class Count2DModel(FeatureBasedBiasModel):
    name = "count_2d"
    format = "mc"

    _choice_dist_cols = [
        f"choice_{i}_dist_from_obj_mean"
        for i in range(6)  # Support up to 6 choices
    ]
    _choice_dist_from_global_cols = [
        f"choice_{i}_dist_from_global_mean"
        for i in range(6)  # Support up to 6 choices
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
        qdf["object"] = qdf["question"].str.extract(r"How many (.*?) are in the image")[0].str.strip()

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
        self.obj_stats = (
            train_df.groupby("object")
            .agg(
                obj_count=("idx", "count"),
                obj_val_mean=("ground_truth", "mean"),
                obj_val_std=("ground_truth", "std"),
                obj_val_log_mean=("log_ground_truth", "mean"),
                obj_val_log_std=("log_ground_truth", "std"),
            )
            .reset_index()
        )

        # Handle std=0 cases
        epsilon = 1e-6
        self.obj_stats["obj_val_std"] = self.obj_stats["obj_val_std"].fillna(0)
        self.obj_stats["obj_val_log_std"] = self.obj_stats["obj_val_log_std"].fillna(0)

        # Calculate ratios
        self.obj_stats["obj_val_log_ratio"] = (
            self.obj_stats["obj_val_log_std"] / (self.obj_stats["obj_val_log_mean"] + epsilon)
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

        # Initialize choice distance columns with NaN
        max_choices = df["n_options"].max()
        for i in range(max_choices):  # Support up to 6 choices
            df[f"choice_{i}_dist_from_obj_mean"] = np.nan
            df[f"choice_{i}_dist_from_global_mean"] = np.nan

        # For each row, calculate distances for each choice
        for row_idx, row in df.iterrows():
            choices = row["choices"]
            n_choices = len(choices)

            # Get the object's mean for this row
            obj_mean = row["obj_val_log_mean"]

            # Calculate distances for each choice
            for choice_idx in range(n_choices):
                choice_val = pd.to_numeric(choices[choice_idx], errors="coerce")
                if pd.notna(choice_val):
                    choice_log = np.log10(choice_val + 1.0)

                    # Calculate distances
                    df.loc[row_idx, f"choice_{choice_idx}_dist_from_obj_mean"] = abs(choice_log - obj_mean)
                    df.loc[row_idx, f"choice_{choice_idx}_dist_from_global_mean"] = abs(
                        choice_log - self.global_mean_log
                    )

        # Privileged features
        # ---------------------------------------
        # Calculate distance from object mean score
        gt_norm_dist = abs(df["log_ground_truth"] - df["obj_val_log_mean"]) / (df["obj_val_log_std"] + epsilon)
        df["gt_log_obj_mean_dist_score"] = 1.0 - minmax_scale(gt_norm_dist + epsilon)

        # Calculate global distance score
        gt_global_dist = abs(df["log_ground_truth"] - self.global_mean_log) / (self.global_std_log + epsilon)
        df["gt_global_mean_dist_score"] = 1.0 - minmax_scale(gt_global_dist + epsilon)

        return df


# 2D Relation
class Relation2DModel(FeatureBasedBiasModel):
    name = "relation_2d"
    format = "mc"

    feature_cols = [
        "n_options",
        "object_1",
        "object_2",
        "pair_freq_score",
        "pair_answer_freq_score",
        "contains_left",  # NEW: Question contains "left"
        "contains_right",  # NEW: Question contains "right"
        "contains_above",  # NEW: Question contains "above"
        "contains_below",  # NEW: Question contains "below"
        "contains_front",  # NEW: Question contains "front"
        "contains_behind",  # NEW: Question contains "behind"
        "spatial_keyword_count",  # NEW: Total spatial keywords
        "question_length",  # NEW: Question length
        "is_majority_answer",  # NEW: Is this the most common answer?
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
            question = question.replace(" (annotated by the red box)", "")
            question = question.replace(" (annotated by the blue box)", "")

            # Try to find patterns like "the X and the Y"
            match = re.search(r"the relative positions of the ([^,]+?) and the ([^,]+?)[, ]", question)
            if match:
                return match.group(1).strip(), match.group(2).strip()

            # Fallback: try to find "the X" and "the Y" separately
            matches = re.findall(r"the ([a-zA-Z0-9_ ]+?)[,?\.]", question)
            if len(matches) >= 2:
                return matches[0].strip(), matches[1].strip()
            return None, None

        # Extract object pairs and sort them for consistency
        qdf[["object_1", "object_2"]] = qdf["question"].apply(lambda q: pd.Series(sorted(extract_objects(q))))

        # Drop rows where extraction failed
        qdf.dropna(subset=["object_1", "object_2"], inplace=True)

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies and answer distributions from training data."""
        # Calculate pair frequencies
        pairs = train_df.apply(lambda row: f"{row['object_1']}-{row['object_2']}", axis=1)
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
                ans: count / total for ans, count in self.pair_answer_freq_map[pair].items()
            }

        # NEW: Calculate answer position frequencies
        self.answer_position_freq = train_df["gt_idx"].value_counts(normalize=True).to_dict()

        # NEW: Find majority answer
        self.answer_distribution = train_df["gt_option"].value_counts()
        self.majority_answer = self.answer_distribution.idxmax()

        # Print diagnostic info
        logger.debug("\n[DIAGNOSTIC] Answer distribution in training:")
        logger.debug(self.answer_distribution.sort_index())
        logger.debug(
            f"Majority answer: {self.majority_answer} ({self.answer_distribution[self.majority_answer] / len(train_df) * 100:.1f}%)"
        )

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and spatial features to the dataframe."""
        if self.pair_freq_map is None or self.pair_answer_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1,
        )

        # Calculate answer frequency score for the pair
        df["pair_answer_freq_score"] = df.apply(
            lambda row: self.pair_answer_freq_map.get((row["object_1"], row["object_2"]), {}).get(row["gt_option"], 0),
            axis=1,
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
            "behind": ["behind", "back", "background"],
        }

        for direction, keywords in spatial_keywords.items():
            df[f"contains_{direction}"] = df["question"].apply(lambda q: int(any(kw in q.lower() for kw in keywords)))

        # NEW: Total spatial keywords
        df["spatial_keyword_count"] = sum(df[f"contains_{direction}"] for direction in spatial_keywords)

        # NEW: Question length
        df["question_length"] = df["question"].str.len()

        # NEW: Is majority answer
        df["is_majority_answer"] = (df["gt_option"] == self.majority_answer).astype(int)

        # NEW: Answer entropy (diversity of answers)
        total_answers = sum(self.answer_distribution.values)
        probs = [count / total_answers for count in self.answer_distribution.values]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        df["answer_entropy"] = entropy

        return df


# 3D Depth
class Depth3DModel(FeatureBasedBiasModel):
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
        qdf[["object_1", "object_2"]] = qdf["choices"].apply(lambda x: pd.Series(sorted(x)))

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies and answer distributions from training data."""
        # Calculate pair frequencies
        pairs = train_df.apply(lambda row: f"{row['object_1']}-{row['object_2']}", axis=1)
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
                ans: count / total for ans, count in self.pair_answer_freq_map[pair].items()
            }

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.pair_freq_map is None or self.pair_answer_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1,
        )

        # Calculate answer frequency score for the pair
        df["pair_answer_freq_score"] = df.apply(
            lambda row: self.pair_answer_freq_map.get((row["object_1"], row["object_2"]), {}).get(row["gt_option"], 0),
            axis=1,
        )

        return df


# 3D Distance
class Distance3DModel(FeatureBasedBiasModel):
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
        qdf[["object_1", "object_2"]] = qdf["choices"].apply(lambda x: pd.Series(sorted(x)))

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies and answer distributions from training data."""
        # Calculate pair frequencies
        pairs = train_df.apply(lambda row: f"{row['object_1']}-{row['object_2']}", axis=1)
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
                ans: count / total for ans, count in self.pair_answer_freq_map[pair].items()
            }

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.pair_freq_map is None or self.pair_answer_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1,
        )

        # Calculate answer frequency score for the pair
        df["pair_answer_freq_score"] = df.apply(
            lambda row: self.pair_answer_freq_map.get((row["object_1"], row["object_2"]), {}).get(row["gt_option"], 0),
            axis=1,
        )

        return df
