import pandas as pd
from datasets import load_dataset


def load_data() -> pd.DataFrame:
    """Load and preprocess VSI-Bench data."""
    vsibench = load_dataset("nyu-visionx/VSI-Bench")
    df_full = vsibench["test"].to_pandas()

    # For numerical questions (no options)
    df_full["gt_val"] = df_full["ground_truth"]
    df_full["gt_idx"] = -1

    # For multiple choice questions (with options)
    mc_mask = df_full["options"].notna()
    df_full.loc[mc_mask, "gt_idx"] = df_full.loc[mc_mask, "ground_truth"].apply(
        lambda x: "ABCD".index(x)
    )
    df_full.loc[mc_mask, "gt_val"] = df_full[mc_mask].apply(
        lambda row: row["options"][int(row["gt_idx"])].split(". ")[-1], axis=1
    )

    return df_full
