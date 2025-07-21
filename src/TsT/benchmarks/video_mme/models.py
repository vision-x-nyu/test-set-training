from typing import Dict
import pandas as pd

from ...protocols import QType


# =============================================================================
# MULTIPLE CHOICE QUESTION MODELS ---------------------------------------------
# =============================================================================


# Video-MME General Model
class VideoMMEModel(QType):
    name = "video_mme"
    format = "mc"

    # Option-level feature columns for each of the 4 options (A, B, C, D)
    _option_feature_cols = [
        f"opt_{i}_{feat}"
        for i in range(4)
        for feat in [
            # "is_numeric",
            "numeric_val",
            # "str_val",
            "is_all_caps",
            "len_option_str",
        ]
    ]

    feature_cols = [
        "duration",
        "domain",
        "sub_category",
        "task_type",
        # "domain_freq_score",
        # "sub_category_freq_score",
        # "task_type_freq_score",
        # "duration_freq_score",
        *_option_feature_cols,
    ]

    def __init__(self):
        # frequency maps learned on the training split
        self.domain_freq_map: pd.Series | None = None
        self.sub_category_freq_map: pd.Series | None = None
        self.task_type_freq_map: pd.Series | None = None
        self.duration_freq_map: pd.Series | None = None

    def _extract_option_text(self, option_str: str) -> str:
        """Extract the text content from an option string like 'A. Apples.'"""
        # Remove the prefix (A. B. C. D.) and strip whitespace
        if option_str and len(option_str) >= 3:
            return option_str[3:].strip()
        return option_str

    def _analyze_option(self, option_str: str) -> Dict:
        """Analyze a single option and return feature dictionary."""
        option_text = self._extract_option_text(option_str)

        # Check if numeric
        is_numeric = False
        numeric_val = float("nan")
        try:
            numeric_val = float(option_text)
            is_numeric = True
        except (ValueError, TypeError):
            pass

        # String features
        str_val = option_text

        # Check if all caps (only for alphabetic characters)
        alphabetic_chars = [c for c in option_text if c.isalpha()]
        is_all_caps = len(alphabetic_chars) > 0 and all(
            c.isupper() for c in alphabetic_chars
        )

        # Length
        len_option_str = len(option_text)

        return {
            "is_numeric": int(is_numeric),
            "numeric_val": numeric_val,
            "str_val": str_val,
            "is_all_caps": int(is_all_caps),
            "len_option_str": len_option_str,
        }

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and preprocess Video-MME questions."""
        # For now, treat all Video-MME questions as the same type
        qdf = df.copy()

        # Ensure we have the required columns
        required_cols = ["duration", "domain", "sub_category", "task_type", "answer"]
        missing_cols = [col for col in required_cols if col not in qdf.columns]
        if missing_cols:
            raise ValueError(
                f"Required columns not found in Video-MME dataset: {missing_cols}"
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
        """Add frequency-based and option-level features to the dataframe."""
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

        # Add option-level features for each row
        for row_idx, row in df.iterrows():
            options = row["options"]

            # Analyze each option (A, B, C, D)
            for opt_idx in range(4):
                if opt_idx < len(options):
                    option_str = options[opt_idx]
                    opt_features = self._analyze_option(option_str)

                    # Add features to dataframe
                    for feat_name, feat_value in opt_features.items():
                        df.loc[row_idx, f"opt_{opt_idx}_{feat_name}"] = feat_value
                else:
                    # Handle cases where there are fewer than 4 options
                    for feat_name in [
                        "is_numeric",
                        "numeric_val",
                        "is_str",
                        "str_val",
                        "is_all_caps",
                        "len_option_str",
                    ]:
                        if feat_name == "numeric_val":
                            df.loc[row_idx, f"opt_{opt_idx}_{feat_name}"] = float("nan")
                        elif feat_name == "str_val":
                            df.loc[row_idx, f"opt_{opt_idx}_{feat_name}"] = ""
                        elif feat_name == "len_option_str":
                            df.loc[row_idx, f"opt_{opt_idx}_{feat_name}"] = 0
                        else:
                            df.loc[row_idx, f"opt_{opt_idx}_{feat_name}"] = 0

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
