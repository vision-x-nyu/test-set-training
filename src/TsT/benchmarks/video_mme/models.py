from typing import Dict
from functools import lru_cache

import pandas as pd
import spacy
from joblib import Memory
import numpy as np

from sentence_transformers import SentenceTransformer, util

from ...protocols import QType


memory = Memory(location=".cache", compress=True, verbose=0)


@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load("en_core_web_sm")


@memory.cache
def get_doc(text: str):
    return get_nlp()(text)


@memory.cache
def linguistic_features(question: str, answer: str) -> Dict:
    # q, a = get_doc(question), get_doc(answer)
    nlp = get_nlp()
    q, a = nlp(question), nlp(answer)
    return {
        # lexical overlap
        "unigram_jaccard": len(set(t.lemma_ for t in q) & set(t.lemma_ for t in a))
        / max(1, len(set(t.lemma_ for t in q) | set(t.lemma_ for t in a))),
        # POS‑pattern similarity
        "pos_bigram_overlap": len(
            set(p for p in zip([t.pos_ for t in q][:-1], [t.pos_ for t in q][1:]))
            & set(p for p in zip([t.pos_ for t in a][:-1], [t.pos_ for t in a][1:]))
        ),
        # counts
        "answer_len": len(a),
        "num_digits": sum(t.like_num for t in a),
        "has_color": any(t.ent_type_ == "COLOR" for t in a),
        # # dependency root match
        # "root_match": int(q.root.lemma_ == a.root.lemma_),
    }


# Sentence-transformers model and embedding cache
ST_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_st_model():
    return SentenceTransformer(ST_MODEL_NAME)


@memory.cache
def get_st_embedding(text: str):
    model = get_st_model()
    return model.encode(text, normalize_embeddings=True)


# =============================================================================
# MULTIPLE CHOICE QUESTION MODELS ---------------------------------------------
# =============================================================================


# Video-MME General Model
N_OPT = 4

# Option-level feature names (single source of truth)
OPTION_FEATURE_NAMES = [
    # "is_numeric",  # Not used in _option_feature_cols
    "numeric_val",
    # "str_val",     # Not used in _option_feature_cols
    "is_all_caps",
    "len_option_str",
]

# Sentence-transformers feature names
ST_FEATURE_NAMES = [
    "sim_q",
    "mean_sim_A",
    "combined_score",
]

QUESTION_OPTION_FEATURE_NAMES = [
    # lexical overlap
    "unigram_jaccard",
    # POS‑pattern similarity
    "pos_bigram_overlap",
    # counts
    "answer_len",
    "num_digits",
    "has_color",
    # # dependency root match
    # "root_match",
]

# Option-level feature columns for each of the 4 options (A, B, C, D)
OPTION_FEATURE_COLS = [f"opt_{i}_{feat}" for i in range(N_OPT) for feat in OPTION_FEATURE_NAMES]
ST_FEATURE_COLS = [f"opt_{i}_{feat}" for i in range(N_OPT) for feat in ST_FEATURE_NAMES]
QUESTION_OPTION_FEATURE_COLS = [f"qo_{i}_{feat}" for i in range(N_OPT) for feat in QUESTION_OPTION_FEATURE_NAMES]


FEATURE_COLS = [
    "duration",
    "domain",
    "sub_category",
    "task_type",
    *OPTION_FEATURE_COLS,
    *ST_FEATURE_COLS,
    *QUESTION_OPTION_FEATURE_COLS,
]


class VideoMMEModel(QType):
    name = "video_mme"
    format = "mc"

    feature_cols = FEATURE_COLS

    def __init__(self):
        # frequency maps learned on the training split
        self.domain_freq_map: pd.Series | None = None
        self.sub_category_freq_map: pd.Series | None = None
        self.task_type_freq_map: pd.Series | None = None
        self.duration_freq_map: pd.Series | None = None

    def _analyze_option(self, option_str: str) -> Dict:
        """Analyze a single option and return feature dictionary."""
        # Check if numeric
        is_numeric = False
        numeric_val = float("nan")
        try:
            numeric_val = float(option_str)
            is_numeric = True
        except (ValueError, TypeError):
            pass

        # String features
        str_val = option_str

        # Check if all caps (only for alphabetic characters)
        alphabetic_chars = [c for c in option_str if c.isalpha()]
        is_all_caps = len(alphabetic_chars) > 0 and all(c.isupper() for c in alphabetic_chars)

        # Length
        len_option_str = len(option_str)

        return {
            "is_numeric": int(is_numeric),
            "numeric_val": numeric_val,
            "str_val": str_val,
            "is_all_caps": int(is_all_caps),
            "len_option_str": len_option_str,
        }

    def _analyze_question_option_pair(self, question: str, option: str) -> Dict:
        """Analyze a question-option pair and return feature dictionary."""
        li_feats = linguistic_features(question, option)
        return {
            **li_feats,
        }

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess Video-MME questions."""
        # For now, treat all Video-MME questions as the same type
        qdf = df.copy()

        # Ensure we have the required columns
        required_cols = ["duration", "domain", "sub_category", "task_type", "answer"]
        missing_cols = [col for col in required_cols if col not in qdf.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found in Video-MME dataset: {missing_cols}")

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect frequency statistics from training data."""
        # Calculate domain frequencies
        self.domain_freq_map = train_df["domain"].value_counts(normalize=True)

        # Calculate sub_category frequencies
        self.sub_category_freq_map = train_df["sub_category"].value_counts(normalize=True)

        # Calculate task_type frequencies
        self.task_type_freq_map = train_df["task_type"].value_counts(normalize=True)

        # Calculate duration frequencies
        self.duration_freq_map = train_df["duration"].value_counts(normalize=True)

    def st_similarity_features(self, question: str, options: list[str], lambda_: float = 0.7):
        """Compute sentence-transformers cosine similarity features for a question and its options."""
        q_vec = get_st_embedding(question)
        A = [get_st_embedding(opt) for opt in options]
        A_mat = np.stack(A)
        # Q→A similarity
        sim_q = util.cos_sim(q_vec, A_mat).cpu().numpy().squeeze()  # (k,)
        # A→A similarity
        sim_A = util.cos_sim(A_mat, A_mat).cpu().numpy()  # (k,k)
        np.fill_diagonal(sim_A, 0)
        mean_sim_A = sim_A.mean(axis=1)
        scores = sim_q - lambda_ * mean_sim_A
        return sim_q, mean_sim_A, scores

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based and option-level features to the dataframe."""
        if self.domain_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate frequency scores
        df["domain_freq_score"] = df["domain"].map(self.domain_freq_map).fillna(0)
        df["sub_category_freq_score"] = df["sub_category"].map(self.sub_category_freq_map).fillna(0)
        df["task_type_freq_score"] = df["task_type"].map(self.task_type_freq_map).fillna(0)
        df["duration_freq_score"] = df["duration"].map(self.duration_freq_map).fillna(0)

        for i in range(N_OPT):
            df[f"opt_{i}"] = df["options"].apply(
                lambda x: x[i].split(". ", 1)[-1].strip()
                if len(x) > i and ". " in x[i]
                else (x[i][3:].strip() if len(x[i]) > 3 else x[i])
            )

        # Vectorized option-level features
        for opt_idx in range(N_OPT):
            opt_col = f"opt_{opt_idx}"
            # Apply _analyze_option to each option string in the column
            opt_feats = df[opt_col].apply(self._analyze_option).apply(pd.Series)
            for feat_name in OPTION_FEATURE_NAMES:
                df[f"{opt_col}_{feat_name}"] = opt_feats[feat_name]

        # Vectorized question-option-level features
        for opt_idx in range(N_OPT):
            opt_col = f"opt_{opt_idx}"
            qo_feats = df.apply(
                lambda row: self._analyze_question_option_pair(row["question"], row[opt_col]), axis=1
            ).apply(pd.Series)
            for qo_name in QUESTION_OPTION_FEATURE_NAMES:
                df[f"qo_{opt_idx}_{qo_name}"] = qo_feats[qo_name]

        # Sentence-transformers features (option-level, all options at once)
        def st_feats_row(row):
            options = [row[f"opt_{i}"] for i in range(N_OPT)]
            sim_q, mean_sim_A, scores = self.st_similarity_features(row["question"], options)
            return {
                **{f"opt_{i}_sim_q": float(sim_q[i]) for i in range(N_OPT)},
                **{f"opt_{i}_mean_sim_A": float(mean_sim_A[i]) for i in range(N_OPT)},
                **{f"opt_{i}_combined_score": float(scores[i]) for i in range(N_OPT)},
            }

        st_feats = df.apply(st_feats_row, axis=1).apply(pd.Series)
        for col in st_feats.columns:
            df[col] = st_feats[col]

        return df


class VideoMMEModelSubset(VideoMMEModel):
    def __init__(self, key: str, val: str):
        super().__init__()
        self.key = key
        self.val = val
        self.name = f"{self.val}"

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess Video-MME questions."""
        qdf = df.copy()
        qdf = qdf[qdf[self.key] == self.val]
        return super().select_rows(qdf)


class VideoMMEModelSubsetCombo(VideoMMEModel):
    def __init__(self, key_vals: dict):
        """
        key_vals: dict mapping column names to values to filter on.
        Example: {"domain": "Sports Competition", "duration": "short"}
        """
        super().__init__()
        self.key_vals = key_vals
        # Create a name based on the key-value pairs, e.g. "domain=Sports Competition|duration=short"
        self.name = "|".join(f"{k}={v}" for k, v in self.key_vals.items())

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess Video-MME questions based on multiple key-value pairs."""
        qdf = df.copy()
        for k, v in self.key_vals.items():
            qdf = qdf[qdf[k] == v]
        return super().select_rows(qdf)
