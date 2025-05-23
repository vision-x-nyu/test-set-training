import os
import json
from typing import Protocol, List, Tuple, Dict, Literal

import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, minmax_scale


# =============================================================================
# 0.  DATA LOADING -------------------------------------------------------------
# =============================================================================

vsibench = load_dataset("nyu-visionx/VSI-Bench")
df_full = vsibench["test"].to_pandas()


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

    # TODO:

# OBJECT ABS DISTANCE
class ObjAbsDistModel(QType):
    name = "object_abs_distance"
    format = "num"

    # TODO:

# OBJECT SIZE ESTIMATION
class ObjSizeEstModel(QType):
    name = "object_size_estimation"
    format = "num"

    # TODO:

# ROOM SIZE ESTIMATION
class RoomSizeEstModel(QType):
    name = "room_size_estimation"
    format = "num"

    # TODO:


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
    _rel_regex = (
        r"which of these objects \((.*?), (.*?), (.*?), (.*?)\) is the closest to the (.*?)\?$"
    )

    def __init__(self):
        # frequency maps learned on the training split
        self.gt_counts: pd.Series | None = None
        self.pair_counts: pd.Series | None = None
        self.ord_pair_counts: pd.Series | None = None

    # ---- interface implementations -----------------------------------------

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = df[df["question_type"] == self.name].copy()
        qdf[[
            "object_1",
            "object_2",
            "object_3",
            "object_4",
            "target_object",
        ]] = qdf["question"].str.extract(self._rel_regex)
        qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
        qdf["gt_option"] = qdf.apply(
            lambda r: r["options"][r["gt_idx"]], axis=1
        )
        qdf["gt_object"] = qdf["gt_option"].apply(
            lambda s: s.split(". ")[-1].strip()
        )
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
            df[f"opt_{i}_option_freq"] = df[f"object_{i+1}"].map(self.gt_counts).fillna(0)
            df[f"opt_{i}_tgt_option_pair_freq"] = df["tgt_gt_pair"].map(self.pair_counts).fillna(0)
            df[f"opt_{i}_tgt_option_ord_pair_freq"] = df["tgt_gt_ord_pair"].map(self.ord_pair_counts).fillna(0)

        df["max_option_freq"] = df[[f"opt_{i}_option_freq" for i in range(4)]].max(axis=1)
        df["max_tgt_option_pair_freq"] = df[[
            f"opt_{i}_tgt_option_pair_freq" for i in range(4)
        ]].max(axis=1)
        df["max_tgt_option_ord_pair_freq"] = df[[
            f"opt_{i}_tgt_option_ord_pair_freq" for i in range(4)
        ]].max(axis=1)
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.gt_counts is None:
            raise RuntimeError("fit_feature_maps must be called first")
        return self._add_rel_feats(df)



# RELATIVE DIRECTION
class RelDirModel(QType):
    name = "object_rel_direction"
    format = "mc"

    # TODO:

# ROUTE PLANNING
class RoutePlanningModel(QType):
    name = "route_planning"
    format = "mc"

    # TODO:


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
        # relative scores
        "gt_pos_score",
        "gt_pair_score",
        "gt_comb_pair_score",
        "gt_obj_score",
        "max_distractor_pos_score",
        "max_distractor_pair_score",
        "max_distractor_comb_pair_score",
        "max_distractor_score",
        "relative_bias_pos_score",
        "relative_bias_pair_score",
        "relative_bias_comb_pair_score",
        "relative_bias_score",
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
            qdf[f"gt_obj_{i+1}"] = qdf["gt_option"].apply(
                lambda s, idx=i: s.split(", ")[idx].strip()
            )
        # pre‑parse option sequences as lists
        for i in range(4):
            qdf[f"opt_seq_{i+1}"] = qdf["options"].apply(
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
                zip(train_df[f"gt_obj_{pos}"], train_df[f"gt_obj_{pos+1}"])
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
            scaled = (
                minmax_scale(arr) if len(np.unique(arr)) > 1 else np.ones_like(arr)
            )
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
                seq = row[f"opt_seq_{i+1}"]
                df_copy.at[row.name, f"opt_seq_{i+1}"] = "|".join(seq)  # cast to str

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
            info["relative_bias_comb_pair_score"] = info["gt_comb_pair_score"] - max_d_comb
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
):
    qdf = model.select_rows(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: List[float] = []

    pbar = tqdm(
        enumerate(skf.split(qdf, qdf["ground_truth"]), 1),
        total=n_splits,
        desc=f"[{model.name.upper()}] CV Folds",
    )
    for fold, (tr_idx, te_idx) in pbar:
        tr, te = qdf.iloc[tr_idx].copy(), qdf.iloc[te_idx].copy()

        model.fit_feature_maps(tr)
        tr = model.add_features(tr)
        te = model.add_features(te)

        X_tr, X_te = tr[model.feature_cols].copy(), te[model.feature_cols].copy()
        encode_categoricals(X_tr, X_te)
        y_tr, y_te = tr["ground_truth"], te["ground_truth"]

        est = _make_estimator(model.task, random_state)
        est.fit(X_tr, y_tr)
        scores.append(_score(est, X_te, y_te, model.metric))
        pbar.set_postfix({f"avg_{model.metric}": f"{np.mean(scores):.2%}"})

    mean_acc, std_acc = float(np.mean(scores)), float(np.std(scores))
    if verbose:
        print(f"\n[{model.name.upper()}] Overall {model.metric.upper()}: {mean_acc:.2%} ± {std_acc:.2%} (n_splits={n_splits})")
        print(f"[{model.name.upper()}] Fold {model.metric.upper()}s: {[f'{s:.2%}' for s in scores]}")

    # full‑data importances ---------------------------------------------------
    model.fit_feature_maps(qdf)  # all rows
    full_df = model.add_features(qdf.copy())
    X_full = full_df[model.feature_cols].copy()
    encode_categoricals(X_full, X_full.copy())
    y_full = full_df["ground_truth"]

    clf_full = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf_full.fit(X_full, y_full)
    fi = (
        pd.DataFrame({"feature": model.feature_cols, "importance": clf_full.feature_importances_})
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", "-k", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--random_state", "-s", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed output"
    )
    args = parser.parse_args()

    models = [
        ## NUM
        # ObjCountModel(),  # TODO:
        # ObjAbsDistModel(),  # TODO:
        # ObjSizeEstModel(),  # TODO:
        # RoomSizeEstModel(),  # TODO:

        ## MC
        RelDistanceModel(),
        # RelDirModel(),  # TODO:
        # RoutePlanningModel(),  # TODO:
        ObjOrderModel()
    ]

    for m in models:
        print(f"\n================  {m.name.upper()}  ================")
        evaluate_bias_model(
            m,
            df_full,
            n_splits=args.n_splits,
            random_state=args.random_state,
            verbose=args.verbose,
        )
