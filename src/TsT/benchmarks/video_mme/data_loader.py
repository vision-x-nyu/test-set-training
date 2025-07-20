import pandas as pd
from datasets import load_dataset


def load_data() -> pd.DataFrame:
    """Load and preprocess Video-MME data."""
    video_mme = load_dataset("lmms-lab/Video-MME")
    df_full = video_mme["test"].to_pandas()

    # Video-MME has multiple choice questions with 4 options
    # Extract the ground truth answer index (A=0, B=1, C=2, D=3)
    df_full["gt_idx"] = df_full["answer"].apply(lambda x: "ABCD".index(x))

    # Extract the ground truth answer text
    df_full["gt_val"] = df_full["answer"]

    return df_full
