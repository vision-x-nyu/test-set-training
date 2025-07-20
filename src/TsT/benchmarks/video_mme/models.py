import pandas as pd

from ...protocols import QType


# =============================================================================
# MULTIPLE CHOICE QUESTION MODELS ---------------------------------------------
# =============================================================================


# Video-MME General Model
class VideoMMEModel(QType):
    name = "video_mme"
    format = "mc"

    feature_cols = [
        "duration",
        "domain",
        "sub_category",
        "task_type",
        # "domain_freq_score",
        # "sub_category_freq_score",
        # "task_type_freq_score",
        # "duration_freq_score",
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.domain_freq_map: pd.Series | None = None
        self.sub_category_freq_map: pd.Series | None = None
        self.task_type_freq_map: pd.Series | None = None
        self.duration_freq_map: pd.Series | None = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess Video-MME questions."""
        # For now, treat all Video-MME questions as the same type
        qdf = df.copy()

        # Ensure we have the required columns
        required_cols = [
            "duration",
            "domain",
            "sub_category",
            "videoID",
            "task_type",
            "answer",
        ]
        for col in required_cols:
            if col not in qdf.columns:
                raise ValueError(
                    f"Required column '{col}' not found in Video-MME dataset"
                )

        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect frequency statistics from training data."""
        # Calculate domain frequencies
        self.domain_freq_map = train_df["domain"].value_counts(normalize=True)

        # Calculate sub_category frequencies
        self.sub_category_freq_map = train_df["sub_category"].value_counts(
            normalize=True
        )

        # Calculate task_type frequencies
        self.task_type_freq_map = train_df["task_type"].value_counts(normalize=True)

        # Calculate duration frequencies
        self.duration_freq_map = train_df["duration"].value_counts(normalize=True)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add frequency-based features to the dataframe."""
        if self.domain_freq_map is None:
            raise RuntimeError("fit_feature_maps must be called first")

        df = df.copy()

        # Calculate frequency scores
        df["domain_freq_score"] = df["domain"].map(self.domain_freq_map).fillna(0)
        df["sub_category_freq_score"] = (
            df["sub_category"].map(self.sub_category_freq_map).fillna(0)
        )
        df["task_type_freq_score"] = (
            df["task_type"].map(self.task_type_freq_map).fillna(0)
        )
        df["duration_freq_score"] = df["duration"].map(self.duration_freq_map).fillna(0)

        return df
