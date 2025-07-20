import pandas as pd
from datasets import load_dataset


def load_data() -> pd.DataFrame:
    """Load and preprocess CV-Bench data."""
    cvbench = load_dataset("nyu-visionx/CV-Bench")
    df_full = cvbench["test"].to_pandas()
    df_full["question_type"] = (
        df_full["task"].str.lower() + "_" + df_full["type"].str.lower()
    )
    df_full["gt_idx"] = df_full["answer"].apply(lambda x: ord(x[1]) - ord("A"))
    df_full["gt_option"] = df_full.apply(
        lambda row: row["choices"][row["gt_idx"]], axis=1
    )
    df_full["n_options"] = df_full["choices"].apply(len)

    return df_full
