import pandas as pd
import spacy
from functools import lru_cache
from ...protocols import QType


@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load("en_core_web_sm")


ALL_IMG_TYPES = [
    "Advertisements",
    "Body Scans: MRI, CT scans, and X-rays",
    "Chemical Structures",
    "Comics and Cartoons",
    "DNA Sequences",
    "Diagrams",
    "Geometric Shapes",
    "Historical Timelines",
    "Icons and Symbols",
    "Landscapes",
    "Logos and Branding",
    "Maps",
    "Mathematical Notations",
    "Medical Images",
    "Microscopic Images",
    "Other",
    "Paintings",
    "Pathological Images",
    "Photographs",
    "Plots and Charts",
    "Portraits",
    "Poster",
    "Screenshots",
    "Sculpture",
    "Sheet Music",
    "Sketches and Drafts",
    "Tables",
    "Technical Blueprints",
    "Trees and Graphs",
]
IMG_TYPE_COLS = [f"img_type_{img_type}" for img_type in ALL_IMG_TYPES]

MAX_NUM_OPTIONS = 5
OPTION_FEATURE_NAMES = ["len", "is_all_caps", "lexical_overlap"]
OPTION_FEATURE_COLS = [f"option_{i}_{opt_feat}" for i in range(MAX_NUM_OPTIONS) for opt_feat in OPTION_FEATURE_NAMES]

FEATURE_COLS = [
    "question_type",
    "subfield",
    "topic_difficulty",
    "num_options",
    *IMG_TYPE_COLS,
]


class MMMUMCModel(QType):
    name = "mmmu"
    format = "mc"

    feature_cols = FEATURE_COLS

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df_mc = df[df.question_type == "multiple-choice"]
        return df_mc.copy()

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        # No-op for baseline
        pass

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for img_type in ALL_IMG_TYPES:
            df[f"img_type_{img_type}"] = df["img_type"].apply(lambda row_types: bool(img_type in row_types))

        # nlp = get_nlp()
        # for i in range(4):
        #     df[f"option_{i}_len"] = df["options"].apply(lambda opts: len(opts[i]) if len(opts) > i else 0)
        #     df[f"option_{i}_is_all_caps"] = df["options"].apply(lambda opts: int(opts[i].isupper()) if len(opts) > i else 0)
        #     df[f"option_{i}_lexical_overlap"] = df.apply(
        #         lambda row: len(set(nlp(row["question"])) & set(nlp(row["options"][i]))) if len(row["options"]) > i else 0,
        #         axis=1
        #     )
        return df


class MMMUModelSubset(MMMUMCModel):
    def __init__(self, key: str, val: str):
        super().__init__()
        self.key = key
        self.val = val
        self.name = f"{self.key}={self.val}"

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = super().select_rows(df)
        return qdf[qdf[self.key] == self.val]
