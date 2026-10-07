"""
TsT-RF feature models for CV-Bench, one per question type.

Every feature is computed from the question text, the answer choices, and
statistics fitted on the training folds only. The target is the gold option
index (``gt_idx``).
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from ...core.protocols import FeatureBasedBiasModel

MAX_CHOICES = 6  # CV-Bench questions have at most 6 answer choices


class _CVBModel(FeatureBasedBiasModel):
    """Shared settings for the CV-Bench models (all multiple choice, target = gold option index)."""

    format = "mc"
    default_target_col = "gt_idx"


# =============================================================================
# MULTIPLE CHOICE QUESTION MODELS ---------------------------------------------
# =============================================================================


# 2D Count
class Count2DModel(_CVBModel):
    name = "count_2d"

    _choice_dist_cols = [f"choice_{i}_dist_from_obj_mean" for i in range(MAX_CHOICES)]
    _choice_dist_from_global_cols = [f"choice_{i}_dist_from_global_mean" for i in range(MAX_CHOICES)]

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

        # Numeric value of the gold option (used only to fit statistics on the training folds)
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

        # Initialize choice distance columns with NaN (every column exists even if a fold
        # has no question with the maximum number of choices)
        for i in range(max(MAX_CHOICES, int(df["n_options"].max()))):
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

        return df


# 2D Relation
class Relation2DModel(_CVBModel):
    name = "relation_2d"

    feature_cols = [
        "n_options",
        "object_1",
        "object_2",
        "pair_freq_score",
        "contains_left",  # question mentions "left"
        "contains_right",
        "contains_above",
        "contains_below",
        "contains_front",
        "contains_behind",
        "spatial_keyword_count",  # number of spatial keyword groups mentioned
        "question_length",
    ]

    def __init__(self):
        self.pair_freq_map: pd.Series | None = None

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
        """Collect object pair frequencies from training data."""
        pairs = train_df.apply(lambda row: f"{row['object_1']}-{row['object_2']}", axis=1)
        self.pair_freq_map = pairs.value_counts(normalize=True)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and spatial features to the dataframe."""
        if self.pair_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1,
        )

        # Spatial keyword features
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

        # Total spatial keywords
        df["spatial_keyword_count"] = sum(df[f"contains_{direction}"] for direction in spatial_keywords)

        # Question length
        df["question_length"] = df["question"].str.len()

        return df


# 3D Depth
class Depth3DModel(_CVBModel):
    name = "depth_3d"

    feature_cols = [
        "object_1",
        "object_2",
        "pair_freq_score",
        "n_options",  # Number of choices available
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.pair_freq_map: pd.Series | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess 3D depth questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # For depth questions, the choices themselves are the objects being compared
        # Sort the choices to ensure consistent pairing
        qdf[["object_1", "object_2"]] = qdf["choices"].apply(lambda x: pd.Series(sorted(x)))

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies from training data."""
        pairs = train_df.apply(lambda row: f"{row['object_1']}-{row['object_2']}", axis=1)
        self.pair_freq_map = pairs.value_counts(normalize=True)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.pair_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1,
        )

        return df


# 3D Distance
class Distance3DModel(_CVBModel):
    name = "distance_3d"

    feature_cols = [
        "object_1",
        "object_2",
        "pair_freq_score",
        "n_options",  # Number of choices available
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.pair_freq_map: pd.Series | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess 3D distance questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # For distance questions, the choices themselves are the objects being compared
        # Sort the choices to ensure consistent pairing
        qdf[["object_1", "object_2"]] = qdf["choices"].apply(lambda x: pd.Series(sorted(x)))

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object pair frequencies from training data."""
        pairs = train_df.apply(lambda row: f"{row['object_1']}-{row['object_2']}", axis=1)
        self.pair_freq_map = pairs.value_counts(normalize=True)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.pair_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate pair frequency score
        df["pair_freq_score"] = df.apply(
            lambda row: self.pair_freq_map.get(f"{row['object_1']}-{row['object_2']}", 0),
            axis=1,
        )

        return df
