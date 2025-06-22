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

vsibench = load_dataset("nyu-visionx/VSI-Bench")
df_full = vsibench["test"].to_pandas()


# =============================================================================
# 1.  HELPERS ------------------------------------------------------------------
# =============================================================================


def encode_categoricals(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Dict[str, LabelEncoder]:
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


# =============================================================================
# 2.  MICRO‑PROTOCOL FOR QUESTION TYPES ---------------------------------------
# =============================================================================


class QType(Protocol):
    name: str
    feature_cols: List[str]
    format: Literal["mc", "num"]

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame: ...

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
        if self.format == "mc":
            return "acc"
        elif self.format == "num":
            return "mra"
        else:
            raise ValueError(f"Unknown format: {self.format}")


# =============================================================================
# 3.  QUESTION TYPE IMPLEMENTATIONS -------------------------------------------
# =============================================================================

"""NUM QUESTIONS"""


# OBJECT COUNTING
class ObjCountModel(QType):
    name = "object_counting"
    format = "num"

    feature_cols = [
        "global_mean_log",
        "global_std_log",
        "object",
        "obj_count",
        "obj_val_mean",
        "obj_val_std",
        "obj_val_log_mean",
        "obj_val_log_std",
        # NOTE: the below features leverage privileged gt info. Remove?
        # "combo_count",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_stats: pd.DataFrame | None = None
        self.combo_counts: pd.Series | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess object counting questions."""
        qdf = df[df["question_type"] == self.name].copy()
        qdf["object"] = (
            qdf["question"]
            .str.extract(r"How many (.*?)\(s\) are in this room")[0]
            .str.strip()
        )
        qdf["ground_truth"] = pd.to_numeric(qdf["ground_truth"], errors="coerce")

        qdf.dropna(subset=["object", "ground_truth"], inplace=True)

        # Add log-transformed ground truth
        qdf["log_ground_truth"] = np.log10(qdf["ground_truth"] + 1.0)

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object and object-ground truth pair frequencies from training data."""
        # Calculate object statistics
        self.obj_stats = (
            train_df.groupby("object")
            .agg(
                obj_count=("id", "count"),
                obj_val_mean=("ground_truth", "mean"),
                obj_val_std=("ground_truth", "std"),
                obj_val_log_mean=("log_ground_truth", "mean"),
                obj_val_log_std=("log_ground_truth", "std"),
            )
            .reset_index()
        )

        # Handle std=0 cases
        self.obj_stats["obj_val_std"] = self.obj_stats["obj_val_std"].fillna(0)
        self.obj_stats["obj_val_log_std"] = self.obj_stats["obj_val_log_std"].fillna(0)

        # Calculate combo counts
        self.combo_counts = train_df.groupby(["object", "ground_truth"]).size()

        # Calculate global statistics
        self.global_mean_log = train_df["log_ground_truth"].mean()
        self.global_std_log = train_df["log_ground_truth"].std()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.obj_stats is None or self.combo_counts is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        epsilon = 1e-6

        # Add object statistics
        df = pd.merge(df, self.obj_stats, on="object", how="left")

        # Add object-ground truth pair frequency
        df["combo_count"] = df.apply(
            lambda row: self.combo_counts.get((row["object"], row["ground_truth"]), 0),
            axis=1,
        )

        # Calculate global distance score
        global_dist = abs(df["log_ground_truth"] - self.global_mean_log) / (
            self.global_std_log + epsilon
        )
        df["global_mean_dist_score"] = 1.0 - minmax_scale(global_dist + epsilon)

        df["global_mean_log"] = self.global_mean_log
        df["global_std_log"] = self.global_std_log

        return df


# OBJECT ABS DISTANCE
class ObjAbsDistModel(QType):
    name = "object_abs_distance"
    format = "num"

    feature_cols = [
        "object_pair",
        "pair_freq_score",
        "pair_inv_var_score",
        "pair_count",
        "pair_val_mean_log",
        "pair_val_std_log",
        # "ord_pair_freq_score",
        # "ord_pair_inv_var_score",
        # "ord_pair_count",
        # "ord_pair_val_mean_log",
        # "ord_pair_val_std_log",
        # # NOTE: the below features leverage privileged gt info. Remove?
        # "pair_mean_dist_score",
        # "global_mean_dist_score",
        # "ord_pair_mean_dist_score",
        # "ord_global_mean_dist_score",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.pair_stats: pd.DataFrame | None = None
        self.global_mean_log: float | None = None
        self.global_std_log: float | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess object absolute distance questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # Extract object pairs from questions
        def extract_objects(question, sort=True):
            match = re.search(
                r"between the (.*?) and the (.*?)(?: \(in meters\))?\?$", question
            )
            if match:
                objs = [match.group(1).strip(), match.group(2).strip()]
                if sort:
                    objs = sorted(objs)
                # else, in the given order
                return "_".join(objs)
            return None

        qdf["object_pair"] = qdf["question"].apply(extract_objects)
        qdf["object_pair_ordered"] = qdf["question"].apply(
            partial(extract_objects, sort=False)
        )  # in the given order
        qdf.dropna(subset=["object_pair"], inplace=True)

        # Convert ground truth to numeric
        qdf["ground_truth"] = pd.to_numeric(qdf["ground_truth"], errors="coerce")
        qdf.dropna(subset=["ground_truth"], inplace=True)

        # Add log-transformed ground truth
        qdf["log_ground_truth"] = np.log10(qdf["ground_truth"] + 1.0)

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect pair statistics and global stats from training data."""
        # Calculate pair statistics
        self.pair_stats = (
            train_df.groupby("object_pair")
            .agg(
                pair_count=("id", "count"),
                pair_val_mean_log=("log_ground_truth", "mean"),
                pair_val_std_log=("log_ground_truth", "std"),
            )
            .reset_index()
        )

        # Calculate pair statistics for the given order
        self.pair_stats_ordered = (
            train_df.groupby("object_pair_ordered")
            .agg(
                ord_pair_count=("id", "count"),
                ord_pair_val_mean_log=("log_ground_truth", "mean"),
                ord_pair_val_std_log=("log_ground_truth", "std"),
            )
            .reset_index()
        )

        # Fill NA std values with 0
        self.pair_stats["pair_val_std_log"] = self.pair_stats[
            "pair_val_std_log"
        ].fillna(0)
        self.pair_stats_ordered["ord_pair_val_std_log"] = self.pair_stats_ordered[
            "ord_pair_val_std_log"
        ].fillna(0)

        # Calculate global statistics
        self.global_mean_log = train_df["log_ground_truth"].mean()
        self.global_std_log = train_df["log_ground_truth"].std()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and statistical features to the dataframe."""
        if self.pair_stats is None or self.global_mean_log is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        epsilon = 1e-6

        # Merge with pair statistics
        df = pd.merge(df, self.pair_stats, on="object_pair", how="left")
        df = pd.merge(df, self.pair_stats_ordered, on="object_pair_ordered", how="left")

        # Calculate pair frequency score
        df["pair_freq_score"] = minmax_scale(df["pair_count"])
        df["ord_pair_freq_score"] = minmax_scale(df["ord_pair_count"])

        # Calculate inverse variance score
        ratio_log = (
            df["pair_val_std_log"] / (df["pair_val_mean_log"] + epsilon)
        ).fillna(0)
        df["pair_inv_var_score"] = 1.0 - minmax_scale(ratio_log + epsilon)
        ord_ratio_log = (
            df["ord_pair_val_std_log"] / (df["ord_pair_val_mean_log"] + epsilon)
        ).fillna(0)
        df["ord_pair_inv_var_score"] = 1.0 - minmax_scale(ord_ratio_log + epsilon)

        # Calculate distance from pair mean score
        norm_dist = abs(df["log_ground_truth"] - df["pair_val_mean_log"]) / (
            df["pair_val_std_log"] + epsilon
        )
        df["pair_mean_dist_score"] = 1.0 - minmax_scale(norm_dist + epsilon)
        ord_norm_dist = abs(df["log_ground_truth"] - df["ord_pair_val_mean_log"]) / (
            df["ord_pair_val_std_log"] + epsilon
        )
        df["ord_pair_mean_dist_score"] = 1.0 - minmax_scale(ord_norm_dist + epsilon)

        # Calculate global distance score
        global_dist = abs(df["log_ground_truth"] - self.global_mean_log) / (
            self.global_std_log + epsilon
        )
        df["global_mean_dist_score"] = 1.0 - minmax_scale(global_dist + epsilon)
        ord_global_dist = abs(df["log_ground_truth"] - self.global_mean_log) / (
            self.global_std_log + epsilon
        )
        df["ord_global_mean_dist_score"] = 1.0 - minmax_scale(ord_global_dist + epsilon)

        return df


# OBJECT SIZE ESTIMATION
class ObjSizeEstModel(QType):
    name = "object_size_estimation"
    format = "num"

    feature_cols = [
        "object",
        "obj_count",
        "obj_freq_score",
        "obj_val_log_ratio",
        "obj_val_log_mean",
        "obj_val_log_std",
        "obj_val_log_ratio",
        # # NOTE: the below features leverage privileged gt info. Remove?
        # "log_obj_mean_dist_score",
        # "log_global_mean_dist_score",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_stats: pd.DataFrame | None = None
        self.global_mean_log: float | None = None
        self.global_std_log: float | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess object size estimation questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # Extract object name from question
        qdf["object"] = qdf["question"].str.extract(r"height\) of the (.*?), measured")[
            0
        ]
        qdf.dropna(subset=["object"], inplace=True)

        # Convert ground truth to numeric
        qdf["ground_truth"] = pd.to_numeric(qdf["ground_truth"], errors="coerce")
        qdf.dropna(subset=["ground_truth"], inplace=True)

        # Add log-transformed ground truth
        qdf["log_ground_truth"] = np.log10(qdf["ground_truth"] + 1.0)

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object statistics and global stats from training data."""
        # Calculate object statistics
        self.obj_stats = (
            train_df.groupby("object")
            .agg(
                obj_count=("id", "count"),
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
            self.obj_stats["obj_val_log_std"]
            / (self.obj_stats["obj_val_log_mean"] + epsilon)
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

        # Calculate distance from object mean score
        norm_dist = abs(df["log_ground_truth"] - df["obj_val_log_mean"]) / (
            df["obj_val_log_std"] + epsilon
        )
        df["log_obj_mean_dist_score"] = 1.0 - minmax_scale(norm_dist + epsilon)

        # Calculate global distance score
        global_dist = abs(df["log_ground_truth"] - self.global_mean_log) / (
            self.global_std_log + epsilon
        )
        df["log_global_mean_dist_score"] = 1.0 - minmax_scale(global_dist + epsilon)

        return df


# ROOM SIZE ESTIMATION
class RoomSizeEstModel(QType):
    name = "room_size_estimation"
    format = "num"

    feature_cols = [
        "global_mean_log",
        "global_std_log",
        # # NOTE: the below features leverage privileged gt info. Remove?
        # "log_size",
        # "pdf_score",
        # "global_mean_dist_score",
        # "global_std_dist_score",
    ]

    def __init__(self):
        # Statistics learned from training data
        self.shape: float | None = None
        self.loc: float | None = None
        self.scale: float | None = None
        self.global_mean_log: float | None = None
        self.global_std_log: float | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess room size estimation questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # Convert ground truth to numeric
        qdf["ground_truth"] = pd.to_numeric(qdf["ground_truth"], errors="coerce")
        qdf.dropna(subset=["ground_truth"], inplace=True)

        # Add log-transformed ground truth
        qdf["log_size"] = np.log10(qdf["ground_truth"] + 1.0)

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Fit lognormal distribution and collect global statistics from training data."""
        # Fit lognormal distribution
        x = train_df["ground_truth"].values
        self.shape, self.loc, self.scale = lognorm.fit(x, floc=0)  # Fix location to 0

        # Calculate global statistics in log space
        self.global_mean_log = train_df["log_size"].mean()
        self.global_std_log = train_df["log_size"].std()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on lognormal distribution and global statistics."""
        if self.shape is None or self.global_mean_log is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        epsilon = 1e-6

        df["global_mean_log"] = self.global_mean_log
        df["global_std_log"] = self.global_std_log

        ## Privileged gt info:

        # Calculate PDF score (higher for more typical sizes)
        pdf = lognorm.pdf(df["ground_truth"], self.shape, self.loc, self.scale)
        df["pdf_score"] = minmax_scale(pdf + epsilon)

        # Calculate distance from global mean in log space
        norm_dist = abs(df["log_size"] - self.global_mean_log) / (
            self.global_std_log + epsilon
        )
        df["global_mean_dist_score"] = 1.0 - minmax_scale(norm_dist + epsilon)

        # Calculate distance from global std in log space
        std_dist = abs(df["log_size"] - self.global_mean_log) / (
            self.global_std_log + epsilon
        )
        df["global_std_dist_score"] = 1.0 - minmax_scale(std_dist + epsilon)

        return df


"""MC QUESTIONS"""


# OBJECT RELATIVE DISTANCE
class RelDistanceModel(QType):
    name = "object_rel_distance"
    format = "mc"

    feature_cols = [
        "object_1",
        "object_2",
        "object_3",
        "object_4",
        "target_object",
        "max_option_freq",
        "max_tgt_option_pair_freq",
        "max_tgt_option_ord_pair_freq",
        *[
            f"opt_{i}_{s}"
            for i in range(4)
            for s in [
                "option_freq",
                "tgt_option_pair_freq",
                "tgt_option_ord_pair_freq",
            ]
        ],
    ]

    # ── helpers ───────────────────────────────────────────────────────────────
    _rel_regex = r"which of these objects \((.*?), (.*?), (.*?), (.*?)\) is the closest to the (.*?)\?$"

    def __init__(self):
        # frequency maps learned on the training split
        self.gt_counts: pd.Series | None = None
        self.pair_counts: pd.Series | None = None
        self.ord_pair_counts: pd.Series | None = None

    # ---- interface implementations -----------------------------------------

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = df[df["question_type"] == self.name].copy()
        qdf[
            [
                "object_1",
                "object_2",
                "object_3",
                "object_4",
                "target_object",
            ]
        ] = qdf["question"].str.extract(self._rel_regex)
        qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
        qdf["gt_option"] = qdf.apply(lambda r: r["options"][r["gt_idx"]], axis=1)
        qdf["gt_object"] = qdf["gt_option"].apply(lambda s: s.split(". ")[-1].strip())
        qdf["tgt_gt_pair"] = qdf.apply(
            lambda r: "-".join(sorted([r["target_object"], r["gt_object"]])),
            axis=1,
        )
        qdf["tgt_gt_ord_pair"] = qdf.apply(
            lambda r: f"{r['target_object']}-{r['gt_object']}", axis=1
        )
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        self.gt_counts = train_df["gt_object"].value_counts()
        self.pair_counts = train_df["tgt_gt_pair"].value_counts()
        self.ord_pair_counts = train_df["tgt_gt_ord_pair"].value_counts()

    def _add_rel_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for i in range(4):
            df[f"opt_{i}_option_freq"] = (
                df[f"object_{i + 1}"].map(self.gt_counts).fillna(0)
            )
            df[f"opt_{i}_tgt_option_pair_freq"] = (
                df["tgt_gt_pair"].map(self.pair_counts).fillna(0)
            )
            df[f"opt_{i}_tgt_option_ord_pair_freq"] = (
                df["tgt_gt_ord_pair"].map(self.ord_pair_counts).fillna(0)
            )

        df["max_option_freq"] = df[[f"opt_{i}_option_freq" for i in range(4)]].max(
            axis=1
        )
        df["max_tgt_option_pair_freq"] = df[
            [f"opt_{i}_tgt_option_pair_freq" for i in range(4)]
        ].max(axis=1)
        df["max_tgt_option_ord_pair_freq"] = df[
            [f"opt_{i}_tgt_option_ord_pair_freq" for i in range(4)]
        ].max(axis=1)
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.gt_counts is None:
            raise RuntimeError("fit_feature_maps must be called first")
        return self._add_rel_feats(df)


# RELATIVE DIRECTION
class RelDirModel(QType):
    name = "object_rel_direction"
    format = "mc"

    feature_cols = [
        "difficulty",
        "positioning_object",
        "orienting_object",
        "querying_object",
        "obj_freq_score",
        # "pos_ori_obj_pair_freq_score",
        # # "pos_ori_obj_ord_pair_freq_score",
        # "poq_obj_pair_freq_score",
        # # "poq_obj_ord_pair_freq_score",
        # # NOTE: the below features leverage privileged gt info. Remove?
        # "gt_opt_freq_score",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_freq_map: pd.Series | None = None
        self.gt_opt_freq_map: pd.Series | None = None
        self.pos_ori_obj_pair_freq_map: pd.Series | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess relative direction questions."""
        # Handle all three subtypes
        qdf = df[df["question_type"].str.startswith("object_rel_direction")].copy()
        qdf["difficulty"] = qdf["question_type"].str.split("_").str[-1]

        # Extract objects from question
        qdf[["positioning_object", "orienting_object", "querying_object"]] = qdf[
            "question"
        ].str.extract(r"standing by the (.*?) and facing the (.*?), is the (.*?) to")

        # Clean object names
        for col in ["positioning_object", "orienting_object", "querying_object"]:
            qdf[col] = qdf[col].str.strip()

        # Extract ground truth answer
        qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
        qdf["gt_option"] = qdf.apply(
            lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1
        )

        # Drop rows where extraction failed
        qdf.dropna(
            subset=[
                "positioning_object",
                "orienting_object",
                "querying_object",
                "gt_option",
            ],
            inplace=True,
        )

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object frequencies, answer frequencies, and object pair frequencies."""
        # Calculate object frequencies
        all_objects = pd.concat(
            [
                train_df["positioning_object"],
                train_df["orienting_object"],
                train_df["querying_object"],
            ]
        ).dropna()
        self.obj_freq_map = all_objects.value_counts(normalize=True)

        # Calculate answer frequencies
        self.gt_opt_freq_map = train_df["gt_option"].value_counts(normalize=True)
        self.pos_obj_freq_map = train_df["positioning_object"].value_counts(
            normalize=True
        )
        self.orient_obj_freq_map = train_df["orienting_object"].value_counts(
            normalize=True
        )
        self.query_obj_freq_map = train_df["querying_object"].value_counts(
            normalize=True
        )

        # Calculate object pair frequencies (positioning-orienting pairs)
        pos_ori_pairs = train_df.apply(
            lambda row: "-".join(
                sorted([row["positioning_object"], row["orienting_object"]])
            ),
            axis=1,
        )
        self.pos_ori_obj_pair_freq_map = pos_ori_pairs.value_counts(normalize=True)

        poq_pairs = train_df.apply(
            lambda row: "-".join(
                sorted(
                    [
                        row["positioning_object"],
                        row["orienting_object"],
                        row["querying_object"],
                    ]
                )
            ),
            axis=1,
        )
        self.poq_obj_pair_freq_map = poq_pairs.value_counts(normalize=True)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.obj_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate object frequency score (sum of normalized frequencies)
        df["obj_freq_score"] = df.apply(
            lambda row: (
                self.obj_freq_map.get(row["positioning_object"], 0)
                + self.obj_freq_map.get(row["orienting_object"], 0)
                + self.obj_freq_map.get(row["querying_object"], 0)
            ),
            axis=1,
        )

        # Calculate answer frequency score
        df["gt_opt_freq_score"] = df["gt_option"].map(self.gt_opt_freq_map).fillna(0)

        # Calculate object pair frequency score
        # positioning-orienting pairs
        df["pos_ori_obj_pair"] = df.apply(
            lambda row: "-".join(
                sorted([row["positioning_object"], row["orienting_object"]])
            ),
            axis=1,
        )
        df["pos_ori_obj_ord_pair"] = df.apply(
            lambda row: "-".join([row["positioning_object"], row["orienting_object"]]),
            axis=1,
        )

        # positioning-orienting-querying pairs
        df["poq_obj_pair"] = df.apply(
            lambda row: "-".join(
                sorted(
                    [
                        row["positioning_object"],
                        row["orienting_object"],
                        row["querying_object"],
                    ]
                )
            ),
            axis=1,
        )

        df["pos_ori_obj_pair_freq_score"] = (
            df["pos_ori_obj_pair"].map(self.pos_ori_obj_pair_freq_map).fillna(0)
        )

        df["poq_obj_pair_freq_score"] = (
            df["poq_obj_pair"].map(self.poq_obj_pair_freq_map).fillna(0)
        )

        return df


# ROUTE PLANNING
class RoutePlanningModel(QType):
    name = "route_planning"
    format = "mc"

    feature_cols = [
        "beginning_object",
        "facing_object",
        "target_object",
        "num_steps",
        "obj_freq_score",
        "route_freq_score",
        "steps_dist_score",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_freq_map: pd.Series | None = None
        self.route_freq_map: pd.Series | None = None
        self.mean_steps: float | None = None
        self.std_steps: float | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess route planning questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # Extract objects from question
        qdf[["beginning_object", "facing_object", "target_object"]] = qdf[
            "question"
        ].str.extract(
            r"You are a robot beginning (?:at|by) the (.*?) "
            r"(?:facing the|facing to|facing towards the|facing|with your back to the) "
            r"(.*?)\. You want to navigate to the (.*?)\."
        )

        # Clean up object names
        for col in ["beginning_object", "facing_object", "target_object"]:
            qdf[col] = (
                qdf[col]
                .str.replace(r" and$", "", regex=True)
                .str.replace(r"^the ", "", regex=True)
                .str.strip()
            )

        # Extract ground truth route
        qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
        qdf["gt_route_str"] = qdf.apply(
            lambda row: row["options"][row["gt_idx"]].strip(), axis=1
        )

        # Calculate number of steps in route
        qdf["num_steps"] = qdf["gt_route_str"].apply(
            lambda x: len(x.split(",")) if pd.notna(x) else 0
        )

        # Drop rows where extraction failed
        qdf.dropna(
            subset=[
                "beginning_object",
                "facing_object",
                "target_object",
                "gt_route_str",
                "ground_truth",
            ],
            inplace=True,
        )

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object frequencies, route frequencies, and step count statistics."""
        # Calculate object frequencies
        all_objects = pd.concat(
            [
                train_df["beginning_object"],
                train_df["facing_object"],
                train_df["target_object"],
            ]
        ).dropna()
        self.obj_freq_map = all_objects.value_counts(normalize=True)

        # Calculate route string frequencies
        self.route_freq_map = train_df["gt_route_str"].value_counts(normalize=True)

        # Calculate step count statistics
        self.mean_steps = train_df["num_steps"].mean()
        self.std_steps = train_df["num_steps"].std()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and statistical features to the dataframe."""
        if self.obj_freq_map is None or self.route_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        epsilon = 1e-6

        # Calculate object frequency score
        df["obj_freq_score"] = df.apply(
            lambda row: (
                self.obj_freq_map.get(row["beginning_object"], 0)
                + self.obj_freq_map.get(row["facing_object"], 0)
                + self.obj_freq_map.get(row["target_object"], 0)
            ),
            axis=1,
        )

        # Calculate route frequency score
        df["route_freq_score"] = df["gt_route_str"].map(self.route_freq_map).fillna(0)

        # Calculate step count typicality score
        norm_dist = abs(df["num_steps"] - self.mean_steps) / (self.std_steps + epsilon)
        df["steps_dist_score"] = 1.0 - minmax_scale(norm_dist + epsilon)

        return df


# OBJECT APPEARANCE ORDER
class ObjOrderModel(QType):
    name = "obj_appearance_order"
    format = "mc"

    _opt_seq_cols = [f"opt_seq_{i}" for i in range(1, 5)]
    _opt_seq_comp_cols = [
        f"seq_{i}_{comp}score"
        for i in range(4)
        for comp in ["pos_", "pair_", "comb_pair_", ""]
    ]
    feature_cols = [
        *_opt_seq_cols,
        *_opt_seq_comp_cols,
        # # NOTE: the below features leverage privileged gt info. Remove?
        # # relative scores
        # "gt_pos_score",
        # "gt_pair_score",
        # "gt_comb_pair_score",
        # "gt_obj_score",
        # "max_distractor_pos_score",
        # "max_distractor_pair_score",
        # "max_distractor_comb_pair_score",
        # "max_distractor_score",
        # "relative_bias_pos_score",
        # "relative_bias_pair_score",
        # "relative_bias_comb_pair_score",
        # "relative_bias_score",
    ]

    def __init__(self):
        self.norm_pos_map: Dict[Tuple[str, int], float] | None = None
        self.norm_pair_map: Dict[Tuple[str, str], float] | None = None
        self.norm_comb_map: Dict[Tuple[str, str], float] | None = None

    # ---- interface implementations -----------------------------------------

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = df[df["question_type"] == self.name].copy()
        qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
        qdf["gt_option"] = qdf.apply(
            lambda r: r["options"][r["gt_idx"]].split(". ")[-1], axis=1
        )
        # split ground‑truth sequence
        for i in range(4):
            qdf[f"gt_obj_{i + 1}"] = qdf["gt_option"].apply(
                lambda s, idx=i: s.split(", ")[idx].strip()
            )
        # pre‑parse option sequences as lists
        for i in range(4):
            qdf[f"opt_seq_{i + 1}"] = qdf["options"].apply(
                lambda opts, idx=i: opts[idx].split(". ", 1)[1].split(", ")
            )
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        # Position frequencies ------------------------------------------------
        pos_counter: Counter = Counter()
        for pos in range(1, 5):
            counts = train_df[f"gt_obj_{pos}"].value_counts()
            for obj, c in counts.items():
                pos_counter[(obj, pos)] += c

        # Adjacent pairs ------------------------------------------------------
        pair_counter: Counter = Counter()
        for pos in range(1, 4):
            pair_counter.update(
                zip(train_df[f"gt_obj_{pos}"], train_df[f"gt_obj_{pos + 1}"])
            )

        # Combination pairs ---------------------------------------------------
        comb_counter: Counter = Counter()
        for i in range(1, 5):
            for j in range(i + 1, 5):
                comb_counter.update(
                    zip(train_df[f"gt_obj_{i}"], train_df[f"gt_obj_{j}"])
                )

        # min‑max normalise ----------------------------------------------------
        def _scale(counter: Counter) -> Dict:
            if not counter:
                return {}
            arr = np.array(list(counter.values())).reshape(-1, 1)
            scaled = minmax_scale(arr) if len(np.unique(arr)) > 1 else np.ones_like(arr)
            return {k: float(scaled[i][0]) for i, k in enumerate(counter.keys())}

        self.norm_pos_map = _scale(pos_counter)
        self.norm_pair_map = _scale(pair_counter)
        self.norm_comb_map = _scale(comb_counter)

    def _add_order_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.norm_pos_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        bias_infos = []
        df_copy = df.copy()
        for _, row in df.iterrows():
            info: Dict[str, float | str | int] = {"id": row["id"]}
            max_d_pos = max_d_pair = max_d_comb = max_d_score = -np.inf

            for i in range(4):
                seq = row[f"opt_seq_{i + 1}"]
                df_copy.at[row.name, f"opt_seq_{i + 1}"] = "|".join(seq)  # cast to str

                pos_score = pair_score = comb_score = 0.0
                for j, obj in enumerate(seq):
                    pos_score += self.norm_pos_map.get((obj, j + 1), 0.0)
                    if j < len(seq) - 1:
                        pair_score += self.norm_pair_map.get((seq[j], seq[j + 1]), 0.0)
                    for k in range(j + 1, len(seq)):
                        comb_score += self.norm_comb_map.get((seq[j], seq[k]), 0.0)
                score = (pos_score + pair_score + comb_score) / 3.0

                info[f"seq_{i}_pos_score"] = pos_score
                info[f"seq_{i}_pair_score"] = pair_score
                info[f"seq_{i}_comb_pair_score"] = comb_score
                info[f"seq_{i}_score"] = score

                if i == row["gt_idx"]:
                    info["gt_pos_score"] = pos_score
                    info["gt_pair_score"] = pair_score
                    info["gt_comb_pair_score"] = comb_score
                    info["gt_obj_score"] = score
                else:
                    max_d_pos = max(max_d_pos, pos_score)
                    max_d_pair = max(max_d_pair, pair_score)
                    max_d_comb = max(max_d_comb, comb_score)
                    max_d_score = max(max_d_score, score)

            info["max_distractor_pos_score"] = max_d_pos
            info["max_distractor_pair_score"] = max_d_pair
            info["max_distractor_comb_pair_score"] = max_d_comb
            info["max_distractor_score"] = max_d_score

            info["relative_bias_pos_score"] = info["gt_pos_score"] - max_d_pos
            info["relative_bias_pair_score"] = info["gt_pair_score"] - max_d_pair
            info["relative_bias_comb_pair_score"] = (
                info["gt_comb_pair_score"] - max_d_comb
            )
            info["relative_bias_score"] = (
                info["relative_bias_pos_score"]
                + info["relative_bias_pair_score"]
                + info["relative_bias_comb_pair_score"]
            ) / 3.0

            bias_infos.append(info)

        bias_df = pd.DataFrame(bias_infos)
        return df_copy.merge(bias_df, on="id", how="left")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._add_order_feats(df)


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
    target_col: str = "ground_truth",
):
    qdf = model.select_rows(df)
    all_scores = []

    # Show progress bar over repeats
    repeat_pbar = tqdm(
        range(repeats), desc=f"[{model.name.upper()}] Repeats", disable=repeats == 1
    )

    for repeat in repeat_pbar:
        current_seed = random_state + repeat

        # Use appropriate splitter based on task type
        if model.task == "reg":
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
            split_args = (qdf,)
        else:  # classification task
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=current_seed
            )
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

    if verbose:
        print(
            f"\n[{model.name.upper()}] Overall {model.metric.upper()}: {mean_acc:.2%} ± {std_acc:.2%} (n_splits={n_splits}, repeats={repeats})"
        )
        if repeats == 1:
            print(
                f"[{model.name.upper()}] Fold {model.metric.upper()}s: {[f'{s:.2%}' for s in all_scores[0]]}"
            )
        else:
            print(
                f"[{model.name.upper()}] Repeat {model.metric.upper()}s: {[f'{s:.2%}' for s in mean_scores]}"
            )

    # full‑data importances ---------------------------------------------------
    model.fit_feature_maps(qdf)  # all rows
    full_df = model.add_features(qdf.copy())
    X_full = full_df[model.feature_cols].copy()
    encode_categoricals(X_full, X_full.copy())
    y_full = full_df[target_col]

    est_full = _make_estimator(model.task, random_state)
    est_full.fit(X_full, y_full)
    fi = (
        pd.DataFrame(
            {"feature": model.feature_cols, "importance": est_full.feature_importances_}
        )
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


def run_evaluation(
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

    # Create models list once
    models = [
        ## NUM
        ObjCountModel(),
        ObjAbsDistModel(),
        ObjSizeEstModel(),
        RoomSizeEstModel(),
        ## MC
        RelDistanceModel(),
        RelDirModel(),
        RoutePlanningModel(),
        ObjOrderModel(),
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
        all_results.append(
            {
                "Model": m.name,
                "Format": m.format.upper(),
                "Metric": m.metric.upper(),
                "Score": mean_score,
                "± Std": std_score,
                "Feature Importances": fi,
            }
        )

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
    print("\n" * 3 + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(
        summary[["Model", "Format", "Metric", "Score", "± Std"]].to_string(index=False)
    )
    print("=" * 80)
    print(f"OVERALL AVERAGE SCORE: {overall_avg:.1%} ± {overall_std:.1%}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_splits", "-k", type=int, default=5, help="Number of CV splits"
    )
    parser.add_argument(
        "--random_state", "-s", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed output"
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=1,
        help="Number of times to repeat evaluation with different random seeds",
    )
    parser.add_argument(
        "--question_types",
        "-q",
        type=str,
        default=None,
        help="Comma-separated list of question types to evaluate (e.g. 'object_counting,object_abs_distance')",
    )
    parser.add_argument(
        "--target_col",
        "-t",
        type=str,
        default="ground_truth",
        help="Column to use as target variable (default: ground_truth)",
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
        target_col=args.target_col,
    )
