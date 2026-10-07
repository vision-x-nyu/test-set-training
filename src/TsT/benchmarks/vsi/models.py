"""
TsT-RF feature models for VSI-Bench, one per question type.

Every feature is computed from the question text, the answer options, and
statistics fitted on the training folds only. The single exception is the
``paper`` preset of RelDistanceModel (see GOLD_PAIR_FEATURES), kept only to
reproduce the proceedings' TsT-RF number.
"""

from typing import Tuple, Dict
import re

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import minmax_scale

from ...core.protocols import FeatureBasedBiasModel

# Feature presets accepted by the VSI-Bench models.
FEATURE_SETS = ("default", "paper")

# object_rel_distance features of the proceedings' TsT-RF (``paper`` preset) that are
# looked up with the held-out question's own gold answer: every opt_{i}_tgt_option_*
# column holds the training frequency of the (target object, *gold* object) pair rather
# than of (target object, option i), and the max_ columns are their maximum. The
# ``default`` preset drops these ten columns.
GOLD_PAIR_FEATURES = (
    "max_tgt_option_pair_freq",
    "max_tgt_option_ord_pair_freq",
    *[f"opt_{i}_tgt_option_pair_freq" for i in range(4)],
    *[f"opt_{i}_tgt_option_ord_pair_freq" for i in range(4)],
)


# =============================================================================
# NUMERICAL QUESTION MODELS ---------------------------------------------------
# =============================================================================


# OBJECT COUNTING
class ObjCountModel(FeatureBasedBiasModel):
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
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_stats: pd.DataFrame | None = None
        self.global_mean_log: float | None = None
        self.global_std_log: float | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess object counting questions."""
        qdf = df[df["question_type"] == self.name].copy()
        qdf["object"] = qdf["question"].str.extract(r"How many (.*?)\(s\) are in this room")[0].str.strip()
        qdf["ground_truth"] = pd.to_numeric(qdf["ground_truth"], errors="coerce")

        qdf.dropna(subset=["object", "ground_truth"], inplace=True)

        # Add log-transformed ground truth
        qdf["log_ground_truth"] = np.log10(qdf["ground_truth"] + 1.0)

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect per-object answer statistics from training data."""
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

        # Calculate global statistics
        self.global_mean_log = train_df["log_ground_truth"].mean()
        self.global_std_log = train_df["log_ground_truth"].std()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.obj_stats is None or self.global_mean_log is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Add object statistics
        df = pd.merge(df, self.obj_stats, on="object", how="left")

        df["global_mean_log"] = self.global_mean_log
        df["global_std_log"] = self.global_std_log

        return df


# OBJECT ABS DISTANCE
class ObjAbsDistModel(FeatureBasedBiasModel):
    name = "object_abs_distance"
    format = "num"

    feature_cols = [
        "object_pair",
        "pair_freq_score",
        "pair_inv_var_score",
        "pair_count",
        "pair_val_mean_log",
        "pair_val_std_log",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.pair_stats: pd.DataFrame | None = None
        self.global_mean_log: float | None = None
        self.global_std_log: float | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess object absolute distance questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # Extract the (sorted) object pair from the question
        def extract_objects(question):
            match = re.search(r"between the (.*?) and the (.*?)(?: \(in meters\))?\?$", question)
            if match:
                return "_".join(sorted([match.group(1).strip(), match.group(2).strip()]))
            return None

        qdf["object_pair"] = qdf["question"].apply(extract_objects)
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

        # Fill NA std values with 0
        self.pair_stats["pair_val_std_log"] = self.pair_stats["pair_val_std_log"].fillna(0)

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

        # Calculate pair frequency score
        df["pair_freq_score"] = minmax_scale(df["pair_count"])

        # Calculate inverse variance score
        ratio_log = (df["pair_val_std_log"] / (df["pair_val_mean_log"] + epsilon)).fillna(0)
        df["pair_inv_var_score"] = 1.0 - minmax_scale(ratio_log + epsilon)

        return df


# OBJECT SIZE ESTIMATION
class ObjSizeEstModel(FeatureBasedBiasModel):
    name = "object_size_estimation"
    format = "num"

    feature_cols = [
        "object",
        "obj_count",
        "obj_freq_score",
        "obj_val_log_mean",
        "obj_val_log_std",
        "obj_val_log_ratio",
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
        qdf["object"] = qdf["question"].str.extract(r"height\) of the (.*?), measured")[0]
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

        return df


# ROOM SIZE ESTIMATION
class RoomSizeEstModel(FeatureBasedBiasModel):
    name = "room_size_estimation"
    format = "num"

    feature_cols = [
        "global_mean_log",
        "global_std_log",
    ]

    def __init__(self):
        # Statistics learned from training data
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
        """Collect global answer statistics (log space) from training data."""
        self.global_mean_log = train_df["log_size"].mean()
        self.global_std_log = train_df["log_size"].std()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the training-set statistics as (constant) features."""
        if self.global_mean_log is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()
        df["global_mean_log"] = self.global_mean_log
        df["global_std_log"] = self.global_std_log
        return df


# =============================================================================
# MULTIPLE CHOICE QUESTION MODELS ---------------------------------------------
# =============================================================================


# OBJECT RELATIVE DISTANCE
class RelDistanceModel(FeatureBasedBiasModel):
    """object_rel_distance: which of four objects is closest to a target object.

    ``default`` (leak-free): the option objects, the target object, and how often
    each option object was the gold answer in the training folds.

    ``paper``: the proceedings' TsT-RF feature list, which adds GOLD_PAIR_FEATURES.
    Those ten columns are looked up with the held-out question's own gold object, so
    the paper preset reads the held-out label. It exists only to reproduce the
    proceedings' number.
    """

    name = "object_rel_distance"
    format = "mc"

    # Proceedings' feature order (the column order changes the Random Forest's
    # feature subsampling, so presets are filtered from this list, never re-ordered).
    _paper_feature_cols = [
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
    feature_cols = [c for c in _paper_feature_cols if c not in GOLD_PAIR_FEATURES]

    # ── helpers ───────────────────────────────────────────────────────────────
    _rel_regex = r"which of these objects \((.*?), (.*?), (.*?), (.*?)\) is the closest to the (.*?)\?$"

    def __init__(self, feature_set: str = "default"):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set '{feature_set}'. Available: {list(FEATURE_SETS)}")
        self.feature_set = feature_set
        self.reads_heldout_label = feature_set == "paper"
        if self.reads_heldout_label:
            self.feature_cols = list(self._paper_feature_cols)
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
        qdf["gt_object"] = qdf["gt_val"]
        if self.reads_heldout_label:
            # (target, gold object) pairs: the paper preset's held-out-label lookup keys
            pairs = list(zip(qdf["target_object"], qdf["gt_object"]))
            qdf["tgt_gt_pair"] = ["-".join(sorted([t, g])) for t, g in pairs]
            qdf["tgt_gt_ord_pair"] = [f"{t}-{g}" for t, g in pairs]
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        # How often each object is the gold answer in the training folds
        self.gt_counts = train_df["gt_object"].value_counts()
        if self.reads_heldout_label:
            self.pair_counts = train_df["tgt_gt_pair"].value_counts()
            self.ord_pair_counts = train_df["tgt_gt_ord_pair"].value_counts()

    def _add_rel_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for i in range(4):
            df[f"opt_{i}_option_freq"] = df[f"object_{i + 1}"].map(self.gt_counts).fillna(0)
            if self.reads_heldout_label:
                # NOTE: keyed on the row's own gold object, not on option i (held-out label)
                df[f"opt_{i}_tgt_option_pair_freq"] = df["tgt_gt_pair"].map(self.pair_counts).fillna(0)
                df[f"opt_{i}_tgt_option_ord_pair_freq"] = df["tgt_gt_ord_pair"].map(self.ord_pair_counts).fillna(0)

        df["max_option_freq"] = df[[f"opt_{i}_option_freq" for i in range(4)]].max(axis=1)
        if self.reads_heldout_label:
            df["max_tgt_option_pair_freq"] = df[[f"opt_{i}_tgt_option_pair_freq" for i in range(4)]].max(axis=1)
            df["max_tgt_option_ord_pair_freq"] = df[[f"opt_{i}_tgt_option_ord_pair_freq" for i in range(4)]].max(axis=1)
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.gt_counts is None:
            raise RuntimeError("fit_feature_maps must be called first")
        return self._add_rel_feats(df)


# RELATIVE DIRECTION
class RelDirModel(FeatureBasedBiasModel):
    name = "object_rel_direction"
    format = "mc"
    # The dataset's three question types, scored together by this one model
    aliases = ("object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard")

    feature_cols = [
        "difficulty",
        "positioning_object",
        "orienting_object",
        "querying_object",
        "obj_freq_score",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_freq_map: pd.Series | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess relative direction questions."""
        # Handle all three subtypes
        qdf = df[df["question_type"].str.startswith("object_rel_direction")].copy()
        qdf["difficulty"] = qdf["question_type"].str.split("_").str[-1]

        # Extract objects from question
        qdf[["positioning_object", "orienting_object", "querying_object"]] = qdf["question"].str.extract(
            r"standing by the (.*?) and facing the (.*?), is the (.*?) to"
        )

        # Clean object names
        for col in ["positioning_object", "orienting_object", "querying_object"]:
            qdf[col] = qdf[col].str.strip()

        # Drop rows where extraction failed
        qdf.dropna(
            subset=[
                "positioning_object",
                "orienting_object",
                "querying_object",
                "gt_val",
            ],
            inplace=True,
        )

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect object frequencies from training data."""
        # Calculate object frequencies
        all_objects = pd.concat(
            [
                train_df["positioning_object"],
                train_df["orienting_object"],
                train_df["querying_object"],
            ]
        ).dropna()
        self.obj_freq_map = all_objects.value_counts(normalize=True)

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

        return df


# ROUTE PLANNING
class RoutePlanningModel(FeatureBasedBiasModel):
    name = "route_planning"
    format = "mc"

    _opt_cols = [f"opt_{i}" for i in range(4)]
    _opt_freq_cols = [f"opt_{i}_freq_score" for i in range(4)]

    feature_cols = [
        "beginning_object",
        "facing_object",
        "target_object",
        "num_steps",
        "num_choices",
        "obj_freq_score",
        *_opt_freq_cols,
        *_opt_cols,
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.obj_freq_map: pd.Series | None = None
        self.route_freq_map: pd.Series | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess route planning questions."""
        qdf = df[df["question_type"] == self.name].copy()

        # Extract objects from question
        qdf[["beginning_object", "facing_object", "target_object"]] = qdf["question"].str.extract(
            r"You are a robot beginning (?:at|by) the (.*?) "
            r"(?:facing the|facing to|facing towards the|facing|with your back to the) "
            r"(.*?)\. You want to navigate to the (.*?)\."
        )

        # Clean up object names
        for col in ["beginning_object", "facing_object", "target_object"]:
            qdf[col] = qdf[col].str.replace(r" and$", "", regex=True).str.replace(r"^the ", "", regex=True).str.strip()

        # Gold route (used only to fit route frequencies on the training folds)
        qdf["gt_route_str"] = qdf["gt_val"]

        qdf["num_choices"] = qdf["options"].apply(lambda x: len(x))

        # get route options
        for i in range(4):
            qdf[f"opt_{i}"] = qdf["options"].apply(lambda x: x[i].split(". ")[-1].strip() if len(x) > i else None)

        # Number of turns to fill in; every option of a question has the same count.
        qdf["num_steps"] = qdf["opt_0"].apply(lambda x: len(x.split(",")) if pd.notna(x) else 0)

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
        """Collect object frequencies and gold-route frequencies from training data."""
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

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and statistical features to the dataframe."""
        if self.obj_freq_map is None or self.route_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate object frequency score
        df["obj_freq_score"] = df.apply(
            lambda row: (
                self.obj_freq_map.get(row["beginning_object"], 0)
                + self.obj_freq_map.get(row["facing_object"], 0)
                + self.obj_freq_map.get(row["target_object"], 0)
            ),
            axis=1,
        )

        # add per-option frequency score (how often each option was the gold route in training)
        for i in range(4):
            df[f"opt_{i}_freq_score"] = df[f"opt_{i}"].map(self.route_freq_map).fillna(0)

        return df


# OBJECT APPEARANCE ORDER
class ObjOrderModel(FeatureBasedBiasModel):
    name = "obj_appearance_order"
    format = "mc"
    target_col_override = "gt_idx"

    _opt_seq_cols = [f"opt_seq_{i}" for i in range(1, 5)]
    _opt_seq_comp_cols = [f"seq_{i}_{comp}score" for i in range(4) for comp in ["pos_", "pair_", "comb_pair_", ""]]
    feature_cols = [
        *_opt_seq_cols,
        *_opt_seq_comp_cols,
    ]

    def __init__(self):
        self.norm_pos_map: Dict[Tuple[str, int], float] | None = None
        self.norm_pair_map: Dict[Tuple[str, str], float] | None = None
        self.norm_comb_map: Dict[Tuple[str, str], float] | None = None

    # ---- interface implementations -----------------------------------------

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = df[df["question_type"] == self.name].copy()
        # split ground‑truth sequence
        for i in range(4):
            qdf[f"gt_obj_{i + 1}"] = qdf["gt_val"].apply(lambda s, idx=i: s.split(", ")[idx].strip())
        # pre‑parse option sequences as lists
        for i in range(4):
            qdf[f"opt_seq_{i + 1}"] = qdf["options"].apply(lambda opts, idx=i: opts[idx].split(". ", 1)[1].split(", "))
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
            pair_counter.update(zip(train_df[f"gt_obj_{pos}"], train_df[f"gt_obj_{pos + 1}"]))

        # Combination pairs ---------------------------------------------------
        comb_counter: Counter = Counter()
        for i in range(1, 5):
            for j in range(i + 1, 5):
                comb_counter.update(zip(train_df[f"gt_obj_{i}"], train_df[f"gt_obj_{j}"]))

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

            bias_infos.append(info)

        bias_df = pd.DataFrame(bias_infos)
        return df_copy.merge(bias_df, on="id", how="left")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._add_order_feats(df)
