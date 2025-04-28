from collections import Counter
import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
from datasets import load_dataset
from sklearn.preprocessing import minmax_scale

from functools import lru_cache

# https://github.com/ShailChoksi/text2digits
from text2digits import text2digits
t2d = text2digits.Text2Digits()


# Define question type categories
NUM_QUESTIONS = {
    "object_size_estimation",
    "object_abs_distance",
    "object_counting",
    "room_size_estimation",
}

MC_QUESTIONS = {
    "object_rel_distance",
    "obj_appearance_order",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction_easy",
    "object_rel_direction_v1",
    "object_rel_direction_v2",
    "object_rel_direction_v3",
    "route_planning",
}

### evaluate file

REF_EVAL_DIR = os.path.abspath("./data/ref_evals")
REF_JSONLS = sorted(
    os.path.join(REF_EVAL_DIR, f)
    for f in os.listdir(REF_EVAL_DIR)
    if f.endswith(".jsonl")
)
REF_MODELS = [
    path.split("/")[-1].replace(".jsonl", "")
    for path in REF_JSONLS
    if path.endswith(".jsonl")
]


def get_model_columns(df):
    model_columns = [col for col in df.columns if col in REF_MODELS]
    if not model_columns or len(model_columns) == 0:
        raise ValueError("No model columns found in the DataFrame")
    return model_columns


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def fuzzy_cleanup(pred):
    return str(pred).strip().split(' ')[0].rstrip('.').strip().lower()

def fuzzy_match(pred, target):
    cleaned_pred = fuzzy_cleanup(pred)
    cleaned_target = fuzzy_cleanup(target)
    return float(cleaned_pred == cleaned_target)

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def abs_dist_norm(pred, target):
    try:
        pred = float(pred)
        target = float(target)
    except BaseException as e:
        return float('nan')
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(to_float(pred), to_float(target)) <= 1 - conf_intervs
    return accuracy.mean()

def fuzzy_cleanup_NA(pred):
    if t2d is None:
        raise ImportError("text2digits is not installed. Please install it to use this function.")

    cleaned_pred = pred.strip().lower().rstrip('.').strip()
    converted_pred = t2d.convert(cleaned_pred)
    return to_float(converted_pred.strip().replace(" ", ""))

def fuzzy_MRA(pred, target, start=0.5, end=0.95, interval=0.05):
    cleaned_pred = fuzzy_cleanup_NA(pred)
    cleaned_target = fuzzy_cleanup_NA(target)
    return mean_relative_accuracy(cleaned_pred, cleaned_target, start, end, interval)

def evaluate_file(json_path):
    if 'jsonl' in json_path:
        with open(json_path, 'r') as file:
            data = [json.loads(line) for line in file]
    else:
        with open(json_path, "r") as f:
            data = json.load(f)
        data = data['logs']

    rows = []

    for doc in tqdm(data, total=len(data), desc=f"Evaluating {json_path}"):
        cur_id = doc['doc']['id']

        pred = doc['doc']['prediction']
        target = doc['doc']['ground_truth']
        question_type = doc['doc']['question_type']

        em = fuzzy_match(pred, target)
        try:
            mra = fuzzy_MRA(pred, target)
        except Exception as e:
            mra = 0.0

        if question_type in NUM_QUESTIONS:
            score = mra
        elif question_type in MC_QUESTIONS:
            # handle old format MC questions
            if "mc_answer" in doc['doc']:
                target = doc['doc']['mc_answer']
                em = fuzzy_match(pred, target)
            score = em
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        # sanity check. max is equivalent
        assert score == max(em, mra), f"Score mismatch: {score} != {max(em, mra)}"

        rows.append({
            'id': cur_id,
            'em': em,
            'mra': mra,
            'score': score,
        })

    df = pd.DataFrame(rows)
    df['em'] = df['em'].astype(float)
    df['mra'] = df['mra'].astype(float)
    df['score'] = df['score'].astype(float)
    return df

def evaluate_files(json_paths=REF_JSONLS):
    dfs = [
        evaluate_file(jsonl_file)[["id", "score"]].rename(
            columns={"score": jsonl_file.split("/")[-1].replace(".jsonl", "")}
        )
        for jsonl_file in json_paths
    ]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="id", how="inner")
    return merged

@lru_cache(maxsize=1)
def get_vsi_with_scores(json_paths=REF_JSONLS):
    merged = evaluate_files(json_paths=json_paths)

    print(f"Loaded {len(merged)} files. Loading VSI-Bench...")
    vsibench = load_dataset("nyu-visionx/VSI-Bench")
    df = vsibench["test"].to_pandas()
    df = df.merge(merged, on="id", how="inner")
    return df

### --- Eval/Visualization Functions --- ###

# Function to evaluate models on filtered dataset
def evaluate_models(df, model_columns=None, sort=False):
    """
    Compute scores for each model on the dataset.

    Args:
        df: DataFrame with model scores
        model_columns: List of column names containing model scores

    Returns:
        DataFrame with model scores by question type
    """
    if model_columns is None or len(model_columns) == 0:
        # If no model columns are provided, check if ref eval models are present
        model_columns = get_model_columns(df)

    # Overall scores
    overall_scores = {model: df[model].mean() for model in model_columns}

    # Scores by question type
    type_scores = df.groupby('question_type').apply(
        lambda x: pd.Series({model: x[model].mean() for model in model_columns})
    )

    # Add overall score as a row
    type_scores.loc['overall'] = pd.Series(overall_scores)

    # Sort columns based on overall scores
    if sort:
        type_scores = type_scores[sorted(type_scores.columns, key=lambda col: type_scores.loc['overall', col], reverse=False)]

    if len(df['question_type'].unique()) == 1:
        # If only one question type, remove overall row
        type_scores = type_scores.drop('overall')

    return type_scores * 100  # Convert to percentage

def visualize_model_scores(df, model_columns=None):
    """
    Visualize model scores using a heatmap.

    Args:
        df: DataFrame with model scores
        model_columns: List of column names containing model scores

    Returns:
        matplotlib figure
    """
    scores = evaluate_models(df, model_columns)

    ranks = scores.rank(axis=1, method='min')

    # Plot heatmap
    plt.figure(figsize=(15, 4))
    # use `ranks` for the heatmap colors, but annotate with `scores`
    sns.heatmap(ranks, annot=scores, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.125)
    plt.xlabel("Model")
    plt.ylabel("Question Type")
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    return plt.gcf()

# Function to visualize the impact of debiasing
def visualize_debiasing_impact(original_df, debiased_df, model_columns=None, title=""):
    """
    Visualize the impact of debiasing on model performance.

    Args:
        original_df: Original dataset with model scores
        debiased_df: Debiased dataset with model scores
        model_columns: List of column names containing model scores
        title: Title for the plot
    """
    # Compute scores before and after debiasing
    original_scores = evaluate_models(original_df, model_columns, sort=True)
    debiased_scores = evaluate_models(debiased_df, model_columns, sort=False)

    # sort debiased by the original scores sort order
    #         type_scores = type_scores[sorted(type_scores.columns, key=lambda col: type_scores.loc['overall', col], reverse=False)]

    debiased_scores = debiased_scores[original_scores.columns]

    n_rows = len(original_scores)
    assert n_rows == len(debiased_scores), f"Number of rows in original and debiased scores do not match: {n_rows} != {len(debiased_scores)}"

    # Compute difference
    diff_scores = debiased_scores - original_scores
    # # Sort by the last column (overall score)
    # diff_scores = diff_scores.T.sort_values(by=diff_scores.index[-1], ascending=False).T

    # Plot results
    # fig, axes = plt.subplots(1, 3, figsize=(25, 2.75*n_rows))
    fig, axes = plt.subplots(1, 3, figsize=(30, 2 + 0.5*n_rows))

    text_rot = 40

    # Original scores
    sns.heatmap(original_scores, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Original Scores")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=text_rot, ha='right')

    # Debiased scores
    sns.heatmap(debiased_scores, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Debiased Scores")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=text_rot, ha='right')

    # Difference
    sns.heatmap(diff_scores, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=axes[2])
    axes[2].set_title("Difference (Debiased - Original)")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=text_rot, ha='right')

    # add title
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=0.95)  # Adjust top to make space for title

    plt.tight_layout()
    return fig

### --- Per-Question Debiasing Functions --- ###

# Functions to identify questions to eliminate for each question type
def filter_object_size_estimation_v1(df, budget):
    """
    Identify object_size_estimation questions to eliminate.

    Args:
        df: DataFrame with object_size_estimation questions
        budget: Number of questions to eliminate

    Returns:
        List of question IDs to eliminate
    """
    # Extract object from question if not already present
    if 'object' not in df.columns:
        df['object'] = df['question'].str.extract(r'height\) of the (.*), measured')[0]

    # Convert ground_truth to numeric if not already done
    if 'ground_truth_num' not in df.columns:
        df['ground_truth_num'] = pd.to_numeric(df['ground_truth'], errors='coerce')

    # Group by object and compute stats
    obj_stats = df.groupby('object').agg(
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std')
    ).reset_index()

    # Identify overrepresented objects
    overrepresented = obj_stats.sort_values('count', ascending=False)

    # Calculate minimum samples to keep for each object
    # We want to keep a minimum of samples for variety
    MIN_SAMPLES = 3

    # Initialize list of IDs to remove
    remove_ids = []

    # Start with most overrepresented objects
    for _, obj_row in overrepresented.iterrows():
        obj_name = obj_row['object']
        obj_count = obj_row['count']
        obj_mean = obj_row['mean']

        # Skip if we don't have more than the minimum
        if obj_count <= MIN_SAMPLES:
            continue

        # Get all questions for this object
        obj_questions = df[df['object'] == obj_name].copy()

        # Calculate how many to remove (at most obj_count - MIN_SAMPLES)
        can_remove = min(obj_count - MIN_SAMPLES, budget - len(remove_ids))

        if can_remove <= 0:
            continue

        # Sort by closeness to mean (most biased = closest to mean)
        obj_questions['distance_to_mean'] = abs(obj_questions['ground_truth_num'] - obj_mean)

        # Select questions closest to the mean for removal
        questions_to_remove = obj_questions.nsmallest(can_remove, 'distance_to_mean')
        remove_ids.extend(questions_to_remove['id'].tolist())

        # Break if we've reached our budget
        if len(remove_ids) >= budget:
            break

    return remove_ids[:budget]  # Ensure we don't exceed budget

def filter_object_size_estimation_hybrid(
    df: pd.DataFrame,
    budget: int,
    min_samples: int = 5,         # Minimum samples to keep per object
    ratio_thresh: float = 0.5,    # Threshold for std/mean ratio
    max_samples_low_var: int = 10, # Max samples if ratio <= ratio_thresh
    max_samples_high_var: int = 15 # Max samples if ratio > ratio_thresh
) -> list:
    """
    Identify object_size_estimation questions to eliminate using a hybrid approach.

    Combines variance-based capping (Jihan's logic) with frequency-based
    prioritization and an overall budget.

    Args:
        df: DataFrame with object_size_estimation questions. Must contain
            'id', 'question', 'ground_truth' columns.
        budget: Total number of questions to eliminate.
        min_samples: Minimum number of samples to keep for any object category.
        ratio_thresh: std/mean ratio threshold to determine variance category.
        max_samples_low_var: Target max samples for low variance objects.
        max_samples_high_var: Target max samples for high variance objects.

    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = df[df['question_type'] == 'object_size_estimation'].copy()

    # --- 1. Preprocessing and Calculate Stats ---
    # Extract object from question if not already present
    # Confirmed regex works for this specific dataset format
    if 'object' not in df_filtered.columns:
        # Use regex to capture object name between 'the ' and ', measured'
        df_filtered['object'] = df_filtered['question'].str.extract(
            r'height\) of the (.*?), measured')[0]
        # Handle potential extraction failures (though unlikely given format)
        df_filtered.dropna(subset=['object'], inplace=True)

    # Convert ground_truth to numeric
    if 'ground_truth_num' not in df_filtered.columns:
        df_filtered['ground_truth_num'] = pd.to_numeric(
            df_filtered['ground_truth'], errors='coerce'
        )
        # Drop rows where conversion failed
        df_filtered.dropna(subset=['ground_truth_num'], inplace=True)

    # Group by object and compute stats
    obj_stats = df_filtered.groupby('object').agg(
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std')
    ).reset_index()

    # --- 2. Determine Target Cap per Object ---
    obj_stats['ratio'] = (obj_stats['std'] / obj_stats['mean']).fillna(0)
    obj_stats['target_max_samples'] = np.where(
        obj_stats['ratio'] <= ratio_thresh,
        max_samples_low_var,
        max_samples_high_var
    )
    # Ensure target_max respects min_samples
    obj_stats['target_max_samples'] = obj_stats['target_max_samples'].clip(lower=min_samples)

    # --- 3. Calculate Potential Removals ---
    # Ensure potential removals doesn't drop below min_samples actual count
    obj_stats['potential_removals'] = (obj_stats['count'] - obj_stats['target_max_samples']).clip(lower=0)
    # Also cap potential removals by how many we *can* remove (count - min_samples)
    obj_stats['potential_removals'] = np.minimum(obj_stats['potential_removals'], (obj_stats['count'] - min_samples).clip(lower=0))

    # Identify objects eligible for removal (must have potential_removals > 0)
    eligible_objects = obj_stats[obj_stats['potential_removals'] > 0].copy()

    # --- 4. Prioritize Removals within Budget ---
    # Prioritize by frequency (descending count) among eligible objects
    eligible_objects.sort_values('count', ascending=False, inplace=True)

    remove_ids = []
    remaining_budget = budget

    for _, obj_row in eligible_objects.iterrows():
        if remaining_budget <= 0:
            break

        obj_name = obj_row['object']
        obj_mean = obj_row['mean']
        # How many *can* we remove from this object based on its potential & budget?
        num_to_remove_from_obj = min(int(obj_row['potential_removals']), remaining_budget)

        if num_to_remove_from_obj <= 0:
            continue

        # Get all questions for this object
        obj_questions = df_filtered[df_filtered['object'] == obj_name].copy()

        # Calculate distance to mean for sorting
        obj_questions['distance_to_mean'] = abs(obj_questions['ground_truth_num'] - obj_mean)

        # Select questions closest to the mean for removal
        questions_to_remove = obj_questions.nsmallest(num_to_remove_from_obj, 'distance_to_mean')

        # Add IDs to removal list and update budget
        ids_found = questions_to_remove['id'].tolist()
        remove_ids.extend(ids_found)
        remaining_budget -= len(ids_found)

    return remove_ids[:budget] # Ensure we strictly adhere to the overall budget


def filter_object_size_estimation_scored(
    df: pd.DataFrame,
    budget: int,
    min_samples: int = 5,         # Minimum samples to keep per object
    w_freq: float = 3.0,          # Weight for frequency contribution
    w_ratio: float = 4.0,         # Weight for inverse variance contribution
    w_dist: float = 5.0           # Weight for closeness to mean contribution
) -> list:
    """
    Identify object_size_estimation questions to eliminate based on a bias score.

    The score combines object frequency, inverse size variance (1/ratio),
    and closeness to the category mean. Higher scores indicate 'worse' samples.

    Args:
        df: DataFrame with object_size_estimation questions. Must contain
            'id', 'question', 'ground_truth' columns.
        budget: Total number of questions to eliminate.
        min_samples: Minimum number of samples to keep for any object category.
        w_freq: Weight for normalized frequency score component.
        w_ratio: Weight for normalized inverse variance score component.
        w_dist: Weight for normalized closeness to mean score component.

    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = df[df['question_type'] == 'object_size_estimation'].copy()

    # --- 1. Preprocessing and Calculate Stats ---
    if 'object' not in df_filtered.columns:
        df_filtered['object'] = df_filtered['question'].str.extract(
            r'height\) of the (.*?), measured')[0]
        df_filtered.dropna(subset=['object'], inplace=True)

    if 'ground_truth_num' not in df_filtered.columns:
        df_filtered['ground_truth_num'] = pd.to_numeric(
            df_filtered['ground_truth'], errors='coerce'
        )
        df_filtered.dropna(subset=['ground_truth_num'], inplace=True)

    # Calculate object stats
    obj_stats = df_filtered.groupby('object').agg(
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std')
    ).reset_index()

    # Handle std=0 cases for ratio calculation
    obj_stats['std'] = obj_stats['std'].fillna(0)
    # Add epsilon to avoid division by zero in ratio and distance normalization
    epsilon = 1e-6
    obj_stats['ratio'] = (obj_stats['std'] / (obj_stats['mean'] + epsilon)).fillna(0)

    # --- 2. Calculate Bias Score Components for each Sample ---
    df_merged = pd.merge(df_filtered, obj_stats, on='object', how='left')

    # Normalize components for scoring (roughly 0-1 range)
    # Frequency Component (higher count = higher score)
    df_merged['freq_score'] = minmax_scale(df_merged['count'])

    # Inverse Variance Component (lower ratio = higher score)
    # Use 1 - normalized_ratio. Add epsilon to ratio before normalization
    df_merged['inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio'] + epsilon)

    # Closeness to Mean Component (closer = higher score)
    # Normalize distance by std dev, then invert (1 - normalized_distance)
    df_merged['norm_dist_from_mean'] = (
        abs(df_merged['ground_truth_num'] - df_merged['mean']) /
        (df_merged['std'] + epsilon)
    )
    # Scale normalized distance to 0-1, then invert
    df_merged['obj_mean_dist_score'] = 1.0 - minmax_scale(df_merged['norm_dist_from_mean'])


    # --- 3. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_freq * df_merged['freq_score'] +
        w_ratio * df_merged['inv_var_score'] +
        w_dist * df_merged['obj_mean_dist_score']
    )

    # --- 4. Filter Based on Score, Respecting Min Samples ---
    # Sort samples by bias score (highest first)
    df_sorted = df_merged.sort_values('bias_score', ascending=False)

    remove_ids = []
    kept_counts = df_sorted.groupby('object')['id'].count().to_dict() # Initial counts

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        obj_name = row['object']

        # Check if removing this sample respects min_samples constraint
        if kept_counts.get(obj_name, 0) > min_samples:
            remove_ids.append(row['id'])
            kept_counts[obj_name] -= 1 # Decrement count for this category

    assert len(remove_ids) <= budget, f"Exceeded budget in filtering process: {len(remove_ids)} > {budget}"

    return remove_ids

def get_object_size_estimation_scores_v2(
    df: pd.DataFrame,
    w_freq: float = 0.0,          # Weight for frequency contribution
    w_ratio: float = 1.0,         # Weight for inverse variance contribution
    w_dist: float = 1.0,          # Weight for closeness to mean contribution
    w_glob_dist: float = 0.0,     # Weight for global mean distance
    logscale: bool = True,       # Use log scale for distance calculation
) -> pd.DataFrame:
    """
    Identify object_size_estimation questions to eliminate based on a bias score.

    The score combines object frequency, inverse size variance (1/ratio),
    and closeness to the category mean. Higher scores indicate 'worse' samples.

    Args:
        df: DataFrame with object_size_estimation questions. Must contain
            'id', 'question', 'ground_truth' columns.
        budget: Total number of questions to eliminate.
        min_samples: Minimum number of samples to keep for any object category.
        w_freq: Weight for normalized frequency score component.
        w_ratio: Weight for normalized inverse variance score component.
        w_dist: Weight for normalized closeness to mean score component.
        logscale: Use log scale for distance calculation

    Returns:
        DataFrame with bias scores and other relevant columns.
    """
    qdf = df[df['question_type'] == 'object_size_estimation'].copy()

    # --- 1. Preprocessing and Calculate Stats ---
    qdf['object'] = qdf['question'].str.extract(
        r'height\) of the (.*?), measured')[0]
    qdf.dropna(subset=['object'], inplace=True)

    qdf['ground_truth_num'] = pd.to_numeric(
        qdf['ground_truth'], errors='coerce'
    )
    qdf.dropna(subset=['ground_truth_num'], inplace=True)

    qdf['log_ground_truth'] = np.log(qdf['ground_truth_num'] + 1e-6) # Add epsilon to avoid log(0)

    # Calculate object stats
    obj_stats = qdf.groupby('object').agg(
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std'),
        log_mean=('log_ground_truth', 'mean'),
        log_std=('log_ground_truth', 'std')
    ).reset_index()

    # Handle std=0 cases for ratio calculation
    epsilon = 1e-6
    obj_stats['std'] = obj_stats['std'].fillna(0)
    obj_stats['log_std'] = obj_stats['log_std'].fillna(0)
    # Add epsilon to avoid division by zero in ratio and distance normalization
    obj_stats['ratio'] = (obj_stats['std'] / (obj_stats['mean'] + epsilon)).fillna(0)
    obj_stats['log_ratio'] = (obj_stats['log_std'] / (obj_stats['log_mean'] + epsilon)).fillna(0)

    # --- 2. Calculate Bias Score Components for each Sample ---
    df_merged = pd.merge(qdf, obj_stats, on='object', how='left')

    # Normalize components for scoring (roughly 0-1 range)
    # Frequency Component (higher count = higher score)
    df_merged['freq_score'] = minmax_scale(df_merged['count'])

    # Inverse Variance Component (lower ratio = higher score)
    # Use 1 - normalized_ratio. Add epsilon to ratio before normalization
    df_merged['inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio'] + epsilon)
    df_merged['log_inv_var_score'] = 1.0 - minmax_scale(df_merged['log_ratio'] + epsilon)

    # Closeness to Mean Component (closer = higher score)
    # Normalize distance by std dev, then invert (1 - normalized_distance)
    df_merged['norm_dist_from_mean'] = (
        abs(df_merged['ground_truth_num'] - df_merged['mean']) /
        (df_merged['std'] + epsilon)
    )
    df_merged['log_norm_dist_from_mean'] = (
        abs(df_merged['log_ground_truth'] - df_merged['log_mean']) /
        (df_merged['log_std'] + epsilon)
    )

    # Scale normalized distance to 0-1, then invert
    df_merged['obj_mean_dist_score'] = 1.0 - minmax_scale(df_merged['norm_dist_from_mean'])
    df_merged['log_obj_mean_dist_score'] = 1.0 - minmax_scale(df_merged['log_norm_dist_from_mean'])

    # global mean dist
    global_mean = df_merged['ground_truth_num'].mean()
    log_global_mean = df_merged['log_ground_truth'].mean()
    df_merged['global_mean_dist'] = abs(df_merged['ground_truth_num'] - global_mean)
    df_merged['log_global_mean_dist'] = abs(df_merged['log_ground_truth'] - log_global_mean)

    # --- 3. Calculate Final Bias Score ---
    if logscale:
        df_merged['bias_score'] = (
            w_freq * df_merged['freq_score'] +
            w_ratio * df_merged['log_inv_var_score'] +
            w_dist * df_merged['log_obj_mean_dist_score'] +
            w_glob_dist * df_merged['log_global_mean_dist']
        )

    else:
        df_merged['bias_score'] = (
            w_freq * df_merged['freq_score'] +
            w_ratio * df_merged['inv_var_score'] +
            w_dist * df_merged['obj_mean_dist_score'] +
            w_glob_dist * df_merged['global_mean_dist']
    )

    return df_merged

def filter_object_size_estimation_scored_v2(
    df: pd.DataFrame,
    budget: int,
    min_samples: int = 5,         # Minimum samples to keep per object
    w_freq: float = 0.0,          # Weight for frequency contribution
    w_ratio: float = 1.0,         # Weight for inverse variance contribution
    w_dist: float = 1.0,          # Weight for closeness to mean contribution
    w_glob_dist: float = 0.0,     # Weight for global mean distance
    logscale: bool = True,       # Use log scale for distance calculation
) -> list:
    """
    Identify object_size_estimation questions to eliminate based on a bias score.

    The score combines object frequency, inverse size variance (1/ratio),
    and closeness to the category mean. Higher scores indicate 'worse' samples.

    Args:
        df: DataFrame with object_size_estimation questions. Must contain
            'id', 'question', 'ground_truth' columns.
        budget: Total number of questions to eliminate.
        min_samples: Minimum number of samples to keep for any object category.
        w_freq: Weight for normalized frequency score component.
        w_ratio: Weight for normalized inverse variance score component.
        w_dist: Weight for normalized closeness to mean score component.
        logscale: Use log scale for distance calculation

    Returns:
        List of question IDs to eliminate.
    """
    df_merged = get_object_size_estimation_scores_v2(df, w_freq=w_freq, w_ratio=w_ratio, w_dist=w_dist, w_glob_dist=w_glob_dist, logscale=logscale)

    # --- 4. Filter Based on Score, Respecting Min Samples ---
    # Sort samples by bias score (highest first)
    df_sorted = df_merged.sort_values('bias_score', ascending=False)

    remove_ids = []
    kept_counts = df_sorted.groupby('object')['id'].count().to_dict() # Initial counts

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        obj_name = row['object']

        # Check if removing this sample respects min_samples constraint
        if kept_counts.get(obj_name, 0) > min_samples:
            remove_ids.append(row['id'])
            kept_counts[obj_name] -= 1 # Decrement count for this category

    assert len(remove_ids) <= budget, f"Exceeded budget in filtering process: {len(remove_ids)} > {budget}"

    return remove_ids



def filter_object_counting_v1(df, budget):
    """
    Identify object_counting questions to eliminate.

    Args:
        df: DataFrame with object_counting questions
        budget: Number of questions to eliminate

    Returns:
        List of question IDs to eliminate
    """
    # Extract object from the question
    if 'object' not in df.columns:
        df['object'] = df['question'].str.extract(r'How many (.*?)\(s\) are in this room')[0]
        df['object'] = df['object'].str.strip()

    # Create object-answer combinations
    df['combo'] = df['object'] + '_' + df['ground_truth'].astype(str)

    # Count frequency of each combination
    combo_counts = df['combo'].value_counts().reset_index()
    combo_counts.columns = ['combo', 'count']

    # Merge counts back to original dataframe
    df = df.merge(combo_counts, on='combo', how='left')

    # Sort by frequency (descending)
    df_sorted = df.sort_values('count', ascending=False)

    # Initialize list of IDs to remove
    remove_ids = []

    # Iterate through combinations, starting with most frequent
    current_combo = None
    combo_removed = 0
    for _, row in df_sorted.iterrows():
        # If we've moved to a new combination, reset counter
        if row['combo'] != current_combo:
            current_combo = row['combo']
            combo_removed = 0

        # Don't remove too many of the same combination
        # Keep at least 1 of each combination
        if combo_removed < (row['count'] - 1):
            remove_ids.append(row['id'])
            combo_removed += 1

        # Break if we've reached our budget
        if len(remove_ids) >= budget:
            break

    return remove_ids[:budget]  # Ensure we don't exceed budget

def filter_object_counting_scored(
    df: pd.DataFrame,
    budget: int,
    min_samples_per_obj: int = 3,     # Min samples to keep for any object
    min_samples_per_gt: int = 1,      # Min samples to keep for any ground truth value
    w_obj_freq: float = 2.0,          # Weight for object frequency
    w_val_freq: float = 5.0,          # Weight for ground truth value frequency
    w_combo_freq: float = 3.0         # Weight for specific (obj, gt) combo frequency
) -> list:
    """
    Identify object_counting questions to eliminate based on a bias score.

    The score combines object frequency, ground truth answer frequency,
    and specific (object, ground_truth) pair frequency.

    Args:
        df: DataFrame with object_counting questions. Must contain
            'id', 'question', 'ground_truth' columns.
        budget: Total number of questions to eliminate.
        min_samples_per_obj: Min samples to keep for any object
        min_samples_per_gt: Min samples to keep for any ground truth value.
        w_obj_freq: Weight for object frequency
        w_val_freq: Weight for ground truth value frequency
        w_combo_freq: Weight for normalized (object, ground_truth) combo frequency

    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = df[df['question_type'] == 'object_counting'].copy()

    # Ensure ground_truth is treated as string for counting discrete values
    df_filtered['ground_truth_str'] = df_filtered['ground_truth'].astype(str)

    # Extract object from the question
    if 'object' not in df.columns:
        df_filtered['object'] = df_filtered['question'].str.extract(r'How many (.*?)\(s\) are in this room')[0]
        df_filtered['object'] = df_filtered['object'].str.strip()

    # --- 1. Calculate Frequencies ---
    # Object frequency
    obj_counts = df_filtered['object'].value_counts().reset_index()
    obj_counts.columns = ['object', 'obj_count']

    # Ground truth value frequency
    val_counts = df_filtered['ground_truth_str'].value_counts().reset_index()
    val_counts.columns = ['ground_truth_str', 'val_count']

    # Specific (object, ground_truth) combo frequency
    combo_counts = df_filtered.groupby(['object', 'ground_truth_str']).size().reset_index(name='combo_count')

    # --- 2. Merge Frequencies and Normalize ---
    df_merged = pd.merge(df_filtered, obj_counts, on='object', how='left')
    df_merged = pd.merge(df_merged, val_counts, on='ground_truth_str', how='left')
    df_merged = pd.merge(df_merged, combo_counts, on=['object', 'ground_truth_str'], how='left')

    # Fill potential NaNs in counts with 0 if a object/gt/combo only appeared once etc.
    df_merged.fillna({'obj_count': 0, 'val_count': 0, 'combo_count': 0}, inplace=True)

    # Normalize counts (min-max scaling to 0-1)
    df_merged['obj_freq_score'] = minmax_scale(df_merged['obj_count'])
    df_merged['val_freq_score'] = minmax_scale(df_merged['val_count'])
    df_merged['combo_freq_score'] = minmax_scale(df_merged['combo_count'])

    # --- 3. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_obj_freq * df_merged['obj_freq_score'] +
        w_val_freq * df_merged['val_freq_score'] +
        w_combo_freq * df_merged['combo_freq_score']
    )

    # --- 4. Filter Based on Score, Respecting Min Samples ---
    df_sorted = df_merged.sort_values('bias_score', ascending=False)

    remove_ids = []
    # Track kept counts for both object and ground_truth value
    kept_obj_counts = df_sorted['object'].value_counts().to_dict()
    kept_val_counts = df_sorted['ground_truth_str'].value_counts().to_dict()

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        obj = row['object']
        val = row['ground_truth_str']

        # Check if removing this sample respects min_samples constraints
        can_remove_obj = kept_obj_counts.get(obj, 0) > min_samples_per_obj
        can_remove_gt = kept_val_counts.get(val, 0) > min_samples_per_gt

        if can_remove_obj and can_remove_gt:
            remove_ids.append(row['id'])
            kept_obj_counts[obj] -= 1
            kept_val_counts[val] -= 1

    return remove_ids


def get_object_counting_scores_v2(
    df: pd.DataFrame,
    w_obj_freq: float = 1.0,          # Weight for object frequency
    w_val_freq: float = 1.0,          # Weight for ground truth value frequency
    w_combo_freq: float = 1.0,         # Weight for specific (obj, gt) combo frequency
) -> pd.DataFrame:
    """
    Identify object_counting questions to eliminate based on a bias score.

    The score combines object frequency, ground truth answer frequency,
    and specific (object, ground_truth) pair frequency.

    Args:
        df: DataFrame with object_counting questions. Must contain
            'id', 'question', 'ground_truth' columns.
        w_obj_freq: Weight for object frequency
        w_val_freq: Weight for ground truth value frequency
        w_combo_freq: Weight for normalized (object, ground_truth) combo frequency

    Returns:
        DataFrame with bias scores and other relevant columns.
    """
    df_filtered = df[df['question_type'] == 'object_counting'].copy()

    # Ensure ground_truth is treated as string for counting discrete values
    df_filtered['ground_truth_str'] = df_filtered['ground_truth'].astype(str)

    # Extract object from the question
    if 'object' not in df.columns:
        df_filtered['object'] = df_filtered['question'].str.extract(r'How many (.*?)\(s\) are in this room')[0]
        df_filtered['object'] = df_filtered['object'].str.strip()

    # --- 1. Calculate Frequencies ---
    # Object frequency
    obj_counts = df_filtered['object'].value_counts().reset_index()
    obj_counts.columns = ['object', 'obj_count']

    # Ground truth value frequency
    val_counts = df_filtered['ground_truth_str'].value_counts().reset_index()
    val_counts.columns = ['ground_truth_str', 'val_count']

    # Specific (object, ground_truth) combo frequency
    combo_counts = df_filtered.groupby(['object', 'ground_truth_str']).size().reset_index(name='combo_count')

    # --- 2. Merge Frequencies and Normalize ---
    df_merged = pd.merge(df_filtered, obj_counts, on='object', how='left')
    df_merged = pd.merge(df_merged, val_counts, on='ground_truth_str', how='left')
    df_merged = pd.merge(df_merged, combo_counts, on=['object', 'ground_truth_str'], how='left')

    # Fill potential NaNs in counts with 0 if a object/gt/combo only appeared once etc.
    df_merged.fillna({'obj_count': 0, 'val_count': 0, 'combo_count': 0}, inplace=True)

    # Normalize counts (min-max scaling to 0-1)
    df_merged['obj_freq_score'] = minmax_scale(df_merged['obj_count'])
    df_merged['val_freq_score'] = minmax_scale(df_merged['val_count'])
    df_merged['combo_freq_score'] = minmax_scale(df_merged['combo_count'])

    # --- 3. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_obj_freq * df_merged['obj_freq_score'] +
        w_val_freq * df_merged['val_freq_score'] +
        w_combo_freq * df_merged['combo_freq_score']
    )
    return df_merged

def filter_object_counting_scored_v2(
    df: pd.DataFrame,
    budget: int,
    min_samples_per_obj: int = 0,     # Min samples to keep for any object
    min_samples_per_gt: int = 0,      # Min samples to keep for any ground truth value
    w_obj_freq: float = 0.0,          # Weight for object frequency
    w_val_freq: float = 1.0,          # Weight for ground truth value frequency
    w_combo_freq: float = 1.0,        # Weight for specific (obj, gt) combo frequency
) -> list:
    if budget <= 0:
        return []

    df_merged = get_object_counting_scores_v2(df, w_obj_freq=w_obj_freq, w_val_freq=w_val_freq, w_combo_freq=w_combo_freq)

    # --- 4. Filter Based on Score, Respecting Min Samples ---
    df_sorted = df_merged.sort_values('bias_score', ascending=False)

    remove_ids = []
    # Track kept counts for both object and ground_truth value
    kept_obj_counts = df_sorted['object'].value_counts().to_dict()
    kept_val_counts = df_sorted['ground_truth_str'].value_counts().to_dict()

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        obj = row['object']
        val = row['ground_truth_str']

        # Check if removing this sample respects min_samples constraints
        can_remove_obj = kept_obj_counts.get(obj, 0) > min_samples_per_obj
        can_remove_gt = kept_val_counts.get(val, 0) > min_samples_per_gt

        if can_remove_obj and can_remove_gt:
            remove_ids.append(row['id'])
            kept_obj_counts[obj] -= 1
            kept_val_counts[val] -= 1

    return remove_ids

def get_object_counting_score_v3(
    df: pd.DataFrame,
    w_obj_freq: float = 1.0,      # Weight for object frequency
    w_combo_freq: float = 1.0     # Weight for specific (obj, gt) combo frequency
    ) -> pd.DataFrame:
    """
    Calculates a secondary bias score for object_counting questions based on
    object frequency and (object, ground_truth) pair frequency. Also returns
    the ground_truth value itself for primary balancing.

    Args:
        df: DataFrame with object_counting questions.
        w_obj_freq: Weight for object frequency component.
        w_combo_freq: Weight for combo frequency component.

    Returns:
        DataFrame with 'id', 'ground_truth_str', and 'secondary_bias_score'.
    """
    qdf = df[df['question_type'] == 'object_counting'].copy()

    qdf['ground_truth_str'] = qdf['ground_truth'].astype(str)
    qdf['object'] = qdf['question'].str.extract(r'How many (.*?)\(s\) are in this room')[0].str.strip()

    # --- Calculate Frequencies ---
    obj_counts = qdf['object'].value_counts().reset_index()
    # obj_counts /= obj_counts.sum()  # Normalize to frequency
    obj_counts.columns = ['object', 'obj_count']
    combo_counts = qdf.groupby(['object', 'ground_truth_str']).size().reset_index(name='combo_count')
    # combo_counts /= combo_counts.sum()  # Normalize to frequency

    # --- Merge Frequencies ---
    df_merged = pd.merge(qdf, obj_counts, on='object', how='left')
    df_merged = pd.merge(df_merged, combo_counts, on=['object', 'ground_truth_str'], how='left')

    # Fill potential NaNs in counts with 0 if a object/gt/combo only appeared once etc.
    df_merged.fillna({'obj_count': 0, 'combo_count': 0}, inplace=True)

    # --- Calculate Bias Score ---
    # Normalize counts (min-max scaling to 0-1)
    df_merged['obj_freq_score'] = minmax_scale(df_merged['obj_count'])
    df_merged['combo_freq_score'] = minmax_scale(df_merged['combo_count'])

    df_merged['bias_score'] = (
        w_obj_freq * df_merged['obj_freq_score'] +
        w_combo_freq * df_merged['combo_freq_score']
    )
    return df_merged

def filter_object_counting_dynamic_pct(
    df: pd.DataFrame, # Original DataFrame
    budget: int,
    max_pct_threshold: float = 0.20, # Target max percentage for any GT value
    w_obj_freq: float = 1.0,
    w_combo_freq: float = 1.0,
    verbose: bool = False
) -> list:
    """
    Filters object_counting questions by iteratively removing samples from the
    GT value group currently exceeding max_pct_threshold, prioritizing removals
    based on a secondary bias score.

    Args:
        df: The original DataFrame containing object_counting questions.
        budget: The maximum number of questions to remove.
        max_pct_threshold: The maximum allowed percentage for any single GT value.
        w_obj_freq: Weight for object frequency in secondary score.
        w_combo_freq: Weight for combo frequency in secondary score.
        verbose: If True, prints status during filtering.

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_object_counting_score_v3(df, w_obj_freq=w_obj_freq, w_combo_freq=w_combo_freq)

    # Sort potential removals by score once (descending) for efficient lookup
    df_sorted = df_with_scores.sort_values('bias_score', ascending=False)
    # Create a lookup from ID to score and GT value
    # id_to_info = df_sorted.set_index('id')[['ground_truth_str', 'bias_score']].to_dict('index')

    # Group IDs by GT value, ordered by descending score
    ids_by_gt_val_sorted = df_sorted.groupby('ground_truth_str')['id'].apply(list).to_dict()

    remove_ids = set()
    current_counts = df_with_scores['ground_truth_str'].value_counts().to_dict()
    total_remaining = len(df_with_scores)

    if verbose: print(f"Starting counts: {current_counts}")

    while len(remove_ids) < budget:
        if total_remaining <= 1: break # Avoid division by zero if only one sample left

        # Calculate current percentages
        current_percentages = {val: count / total_remaining for val, count in current_counts.items()}

        # Find GT values exceeding the threshold
        eligible_vals = {val: pct for val, pct in current_percentages.items() if pct > max_pct_threshold}

        if not eligible_vals:
            if verbose: print(f"No GT value exceeds threshold {max_pct_threshold:.1%}. Stopping.")
            break # Stop if no value exceeds threshold

        # Find the value with the highest current percentage among eligible ones
        # If tie, max() picks one arbitrarily, which is okay
        val_to_reduce = max(eligible_vals, key=eligible_vals.get)

        # Find the highest-scoring ID for this value *that hasn't been removed yet*
        id_to_remove = None
        if val_to_reduce in ids_by_gt_val_sorted:
            # Iterate through pre-sorted IDs for this value
            for potential_id in ids_by_gt_val_sorted[val_to_reduce]:
                if potential_id not in remove_ids:
                    id_to_remove = potential_id
                    break # Found the highest-scoring available ID

        if id_to_remove is None:
            # Should not happen if current_counts[val_to_reduce] > 0,
            # but maybe indicates an issue or that all samples for this value were removed.
            if verbose: print(f"Warning: Could not find an available ID for GT value {val_to_reduce}. Stopping.")
            break # Stop if we can't find a sample to remove

        # Remove the selected ID
        remove_ids.add(id_to_remove)
        current_counts[val_to_reduce] -= 1
        if current_counts[val_to_reduce] == 0: # Clean up if count reaches zero
            del current_counts[val_to_reduce]
        total_remaining -= 1

        if verbose and (len(remove_ids) % 10 == 0 or len(remove_ids) == budget):
            # print(f"Removed {len(remove_ids)}/{budget}. Reducing '{val_to_reduce}'. New count: {current_counts.get(val_to_reduce, 0)}. Total left: {total_remaining}")
            formatted_percentages = {key: f"{value:.1%}" for key, value in current_percentages.items()}
            print(f"Removed {len(remove_ids)}/{budget}. \tCurrent pcts: {formatted_percentages}")

    if verbose:
        final_percentages = {val: count / total_remaining for val, count in current_counts.items()} if total_remaining > 0 else {}
        print(f"Finished. Removed {len(remove_ids)} samples.")
        print(f"- Final counts:      \t{current_counts}")
        print(f"- Final percentages: \t{ {k: f'{v:.1%}' for k, v in final_percentages.items()} }")

    return list(remove_ids)

def filter_object_abs_distance_v1(df, budget):
    """
    Identify object_abs_distance questions to eliminate.

    Args:
        df: DataFrame with object_abs_distance questions
        budget: Number of questions to eliminate

    Returns:
        List of question IDs to eliminate
    """
    # Extract category pairs from questions
    if 'category_pair' not in df.columns:
        def extract_categories(question):
            match = re.search(r'between the (.*) and the (.*)\?', question)
            if match:
                cats = sorted([match.group(1).strip(), match.group(2).strip()])
                return '_'.join(cats)
            return None

        df['category_pair'] = df['question'].apply(extract_categories)

    # Convert ground_truth to numeric if not already done
    if 'ground_truth_num' not in df.columns:
        df['ground_truth_num'] = pd.to_numeric(df['ground_truth'], errors='coerce')

    # Group by category pair and compute stats
    pair_stats = df.groupby('category_pair').agg(
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std')
    ).reset_index()

    # Identify overrepresented pairs
    overrepresented = pair_stats.sort_values('count', ascending=False)

    # Calculate minimum samples to keep for each pair
    MIN_SAMPLES = 3

    # Initialize list of IDs to remove
    remove_ids = []

    # Start with most overrepresented pairs
    for _, pair_row in overrepresented.iterrows():
        pair_name = pair_row['category_pair']
        pair_count = pair_row['count']
        pair_mean = pair_row['mean']

        # Skip if we don't have more than the minimum
        if pair_count <= MIN_SAMPLES:
            continue

        # Get all questions for this pair
        pair_questions = df[df['category_pair'] == pair_name].copy()

        # Calculate how many to remove (at most pair_count - MIN_SAMPLES)
        can_remove = min(pair_count - MIN_SAMPLES, budget - len(remove_ids))

        if can_remove <= 0:
            continue

        # Sort by closeness to mean
        pair_questions['distance_to_mean'] = abs(pair_questions['ground_truth_num'] - pair_mean)

        # Select questions closest to the mean for removal
        questions_to_remove = pair_questions.nsmallest(can_remove, 'distance_to_mean')
        remove_ids.extend(questions_to_remove['id'].tolist())

        # Break if we've reached our budget
        if len(remove_ids) >= budget:
            break

    return remove_ids[:budget]  # Ensure we don't exceed budget

def get_object_abs_distance_bias(
    df: pd.DataFrame,
    w_pair_freq: float = 1.0,     # Weight for pair frequency component
    w_pair_inv_var: float = 1.0,         # Weight for inverse variance component
    w_pair_dist: float = 1.0           # Weight for closeness to mean component
) -> pd.DataFrame:
    """
    Identify object_abs_distance questions to eliminate based on a bias score.

    The score combines object pair frequency, inverse distance variance (1/ratio),
    and closeness to the pair's mean distance. Higher scores indicate 'worse' samples.

    Args:
        df: DataFrame with object_abs_distance questions. Must contain
            'id', 'question', 'ground_truth' columns.
        w_pair_freq: Weight for normalized pair frequency score component.
        w_pair_inv_var: Weight for normalized inverse variance (low std/mean) component.
        w_pair_dist: Weight for normalized closeness to mean score component.

    Returns:
        DataFrame with bias scores and other relevant columns.
    """

    df_filtered = df[df['question_type'] == 'object_abs_distance'].copy()
    if df_filtered.empty:
        return pd.DataFrame()

    # --- 1. Preprocessing and Calculate Stats ---
    # Extract object pairs from questions
    if 'object_pair' not in df_filtered.columns: # Use object_pair now
        def extract_objects(question):
            # Regex to find objects between "between the " and " and the " and ending with "?" possibly "(in meters)?"
            match = re.search(r'between the (.*?) and the (.*?)(?: \(in meters\))?\?$', question)
            if match:
                # Sort objects alphabetically to treat (A, B) and (B, A) the same
                objs = sorted([match.group(1).strip(), match.group(2).strip()])
                return '_'.join(objs)
            return None
        df_filtered['object_pair'] = df_filtered['question'].apply(extract_objects)
        # Drop rows where pair extraction failed
        df_filtered.dropna(subset=['object_pair'], inplace=True)


    # Convert ground_truth to numeric
    if 'ground_truth_num' not in df_filtered.columns:
        df_filtered['ground_truth_num'] = pd.to_numeric(
            df_filtered['ground_truth'], errors='coerce'
        )
        df_filtered.dropna(subset=['ground_truth_num'], inplace=True) # Drop conversion failures

    # Group by object pair and compute stats
    pair_stats = df_filtered.groupby('object_pair').agg( # Group by object_pair
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std')
    ).reset_index()

    # Handle std=0 or mean=0 cases
    pair_stats['std'] = pair_stats['std'].fillna(0)
    epsilon = 1e-6
    pair_stats['ratio'] = (pair_stats['std'] / (pair_stats['mean'] + epsilon)).fillna(0)

    # --- 2. Calculate Bias Score Components for each Sample ---
    df_merged = pd.merge(df_filtered, pair_stats, on='object_pair', how='left')

    # Normalize components for scoring (roughly 0-1 range)
    # Pair Frequency Component (higher count = higher score)
    df_merged['pair_freq_score'] = minmax_scale(df_merged['count'])

    # Inverse Variance Component (lower ratio = higher score)
    df_merged['pair_inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio'] + epsilon)

    # Closeness to Mean Component (closer = higher score)
    df_merged['norm_dist_from_mean'] = (
        abs(df_merged['ground_truth_num'] - df_merged['mean']) /
        (df_merged['std'] + epsilon)
    )
    df_merged['pair_mean_dist_score'] = 1.0 - minmax_scale(df_merged['norm_dist_from_mean'])

    # --- 3. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_pair_freq * df_merged['pair_freq_score'] +
        w_pair_inv_var * df_merged['pair_inv_var_score'] +
        w_pair_dist * df_merged['pair_mean_dist_score']
    )

    # --- 4. Filter Based on Score, Respecting Min Samples ---
    return df_merged.sort_values('bias_score', ascending=False)

def get_object_abs_distance_bias_v2(
    df: pd.DataFrame,
    w_pair_freq: float = 1.0,     # Weight for pair frequency component
    w_pair_inv_var: float = 1.0,         # Weight for inverse variance component
    w_pair_dist: float = 1.0           # Weight for closeness to mean component
) -> pd.DataFrame:
    """
    Identify object_abs_distance questions to eliminate based on a bias score.

    The score combines object pair frequency, inverse distance variance (1/ratio),
    and closeness to the pair's mean distance. Higher scores indicate 'worse' samples.

    Args:
        df: DataFrame with object_abs_distance questions. Must contain
            'id', 'question', 'ground_truth' columns.
        w_pair_freq: Weight for normalized pair frequency score component.
        w_pair_inv_var: Weight for normalized inverse variance (low std/mean) component.
        w_pair_dist: Weight for normalized closeness to mean score component.

    Returns:
        DataFrame with bias scores and other relevant columns.
    """

    df_filtered = df[df['question_type'] == 'object_abs_distance'].copy()
    if df_filtered.empty:
        return pd.DataFrame()

    # --- 1. Preprocessing and Calculate Stats ---
    # Extract object pairs from questions
    if 'object_pair' not in df_filtered.columns: # Use object_pair now
        def extract_objects(question):
            # Regex to find objects between "between the " and " and the " and ending with "?" possibly "(in meters)?"
            match = re.search(r'between the (.*?) and the (.*?)(?: \(in meters\))?\?$', question)
            if match:
                # Sort objects alphabetically to treat (A, B) and (B, A) the same
                objs = sorted([match.group(1).strip(), match.group(2).strip()])
                return '_'.join(objs)
            return None
        df_filtered['object_pair'] = df_filtered['question'].apply(extract_objects)
        # Drop rows where pair extraction failed
        df_filtered.dropna(subset=['object_pair'], inplace=True)


    # Convert ground_truth to numeric
    if 'ground_truth_num' not in df_filtered.columns:
        df_filtered['ground_truth_num'] = pd.to_numeric(
            df_filtered['ground_truth'], errors='coerce'
        )
        df_filtered.dropna(subset=['ground_truth_num'], inplace=True) # Drop conversion failures

    # Group by object pair and compute stats
    pair_stats = df_filtered.groupby('object_pair').agg( # Group by object_pair
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std')
    ).reset_index()

    # Handle std=0 or mean=0 cases
    pair_stats['std'] = pair_stats['std'].fillna(0)
    epsilon = 1e-6
    pair_stats['ratio'] = (pair_stats['std'] / (pair_stats['mean'] + epsilon)).fillna(0)

    # --- 2. Calculate Bias Score Components for each Sample ---
    df_merged = pd.merge(df_filtered, pair_stats, on='object_pair', how='left')

    # filter out pairs with count 1
    df_ct1 = df_merged.loc[df_merged["count"] == 1]
    df_merged = df_merged.loc[df_merged["count"] > 1]

    # Normalize components for scoring (roughly 0-1 range)
    # Pair Frequency Component (higher count = higher score)
    df_merged['pair_freq_score'] = minmax_scale(df_merged['count'])

    # Inverse Variance Component (lower ratio = higher score)
    df_merged['pair_inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio'] + epsilon)

    # Closeness to Mean Component (closer = higher score)
    df_merged['norm_dist_from_mean'] = (
        abs(df_merged['ground_truth_num'] - df_merged['mean']) /
        (df_merged['std'] + epsilon)
    )
    df_merged['pair_mean_dist_score'] = 1.0 - minmax_scale(df_merged['norm_dist_from_mean'])

    # --- 3. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_pair_freq * df_merged['pair_freq_score'] +
        w_pair_inv_var * df_merged['pair_inv_var_score'] +
        w_pair_dist * df_merged['pair_mean_dist_score']
    )

    # merge back on the count_1 data, fill empty bias scores with 0
    df_merged = pd.concat([df_merged, df_ct1]).fillna(0)


    # --- 4. Filter Based on Score, Respecting Min Samples ---
    return df_merged.sort_values('bias_score', ascending=False)


def filter_object_abs_distance_scored(
    df: pd.DataFrame,
    budget: int,
    min_samples_per_pair: int = 3, # Min samples to keep per object pair
    w_pair_freq: float = 1.0,      # Weight for pair frequency component
    w_pair_inv_var: float = 1.0,        # Weight for inverse variance component
    w_pair_dist: float = 1.0            # Weight for closeness to mean component
) -> list:
    """
    Identify object_abs_distance questions to eliminate based on a bias score.

    The score combines object pair frequency, inverse distance variance (1/ratio),
    and closeness to the pair's mean distance. Higher scores indicate 'worse' samples.

    Args:
        df: DataFrame with object_abs_distance questions. Must contain
            'id', 'question', 'ground_truth' columns.
        budget: Total number of questions to eliminate.
        min_samples_per_pair: Min samples to keep for any object pair.
        w_pair_freq: Weight for normalized pair frequency score component.
        w_pair_inv_var: Weight for normalized inverse variance (low std/mean) component.
        w_pair_dist: Weight for normalized closeness to mean score component.

    Returns:
        List of question IDs to eliminate.
    """

    df_sorted = get_object_abs_distance_bias(df, w_pair_freq, w_pair_inv_var, w_pair_dist)

    remove_ids = []
    kept_counts = df_sorted['object_pair'].value_counts().to_dict()

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        pair_name = row['object_pair']

        # Check if removing this sample respects min_samples constraint
        if kept_counts.get(pair_name, 0) > min_samples_per_pair:
            remove_ids.append(row['id'])
            kept_counts[pair_name] -= 1

    return remove_ids


def get_object_abs_distance_scored_log(
    df: pd.DataFrame,
    w_pair_freq: float = 1.0,     # Weight for pair frequency component
    w_pair_inv_var: float = 1.0,         # Weight for inverse variance (orig ratio) component
    w_pair_dist: float = 1.0       # Weight for closeness to mean in log-space
) -> pd.DataFrame:
    """
    Identify object_abs_distance questions to eliminate based on a bias score,
    using log-transformed distances for closeness calculation.

    The score combines object pair frequency, inverse distance variance (original std/mean),
    and closeness to the pair's mean *log*-distance.

    Args:
        df: DataFrame with object_abs_distance questions.
        w_pair_freq: Weight for normalized pair frequency score component.
        w_pair_inv_var: Weight for normalized inverse variance (low std/mean of original distances).
        w_pair_dist: Weight for normalized closeness to mean in log-space.

    Returns:
        DataFrame with bias scores and other relevant columns.
    """

    df_filtered = df[df['question_type'] == 'object_abs_distance'].copy()
    if df_filtered.empty:
        return []

    # --- 1. Preprocessing, Calculate Stats (Original and Log) ---
    if 'object_pair' not in df_filtered.columns:
        def extract_objects(question):
            match = re.search(r'between the (.*?) and the (.*?)(?: \(in meters\))?\?$', question)
            if match:
                objs = sorted([match.group(1).strip(), match.group(2).strip()])
                return '_'.join(objs)
            return None
        df_filtered['object_pair'] = df_filtered['question'].apply(extract_objects)
        df_filtered.dropna(subset=['object_pair'], inplace=True)

    # Convert ground_truth to numeric
    if 'ground_truth_num' not in df_filtered.columns:
        df_filtered['ground_truth_num'] = pd.to_numeric(
            df_filtered['ground_truth'], errors='coerce'
        )
        df_filtered.dropna(subset=['ground_truth_num'], inplace=True)

    # Add epsilon for log transformation to avoid log(0)
    epsilon = 1e-6
    df_filtered['log_ground_truth_num'] = np.log(df_filtered['ground_truth_num'] + epsilon)

    # Group by object pair and compute stats for both original and log distances
    pair_stats = df_filtered.groupby('object_pair').agg(
        count=('id', 'count'),
        mean_orig=('ground_truth_num', 'mean'),
        std_orig=('ground_truth_num', 'std'),
        mean_log=('log_ground_truth_num', 'mean'),
        std_log=('log_ground_truth_num', 'std')
    ).reset_index()

    # Calculate original std/mean ratio
    pair_stats['std_orig'] = pair_stats['std_orig'].fillna(0)
    pair_stats['ratio_orig'] = (pair_stats['std_orig'] / (pair_stats['mean_orig'] + epsilon)).fillna(0)
    pair_stats['std_log'] = pair_stats['std_log'].fillna(0)


    # --- 2. Calculate Bias Score Components for each Sample ---
    df_merged = pd.merge(df_filtered, pair_stats, on='object_pair', how='left')

    # Normalize components for scoring (roughly 0-1 range)
    # Pair Frequency Component
    df_merged['pair_freq_score'] = minmax_scale(df_merged['count'])

    # Inverse Variance Component (using original std/mean ratio)
    df_merged['pair_inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio_orig'] + epsilon)

    # Closeness to Mean Component (in log-space)
    df_merged['norm_dist_from_pair_mean_log'] = (
        abs(df_merged['log_ground_truth_num'] - df_merged['mean_log']) /
        (df_merged['std_log'] + epsilon)
    )
    # Add small value before scaling in case all distances are identical (std_log=0)
    df_merged['pair_mean_dist_score'] = 1.0 - minmax_scale(df_merged['norm_dist_from_pair_mean_log'] + epsilon)


    # --- 3. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_pair_freq * df_merged['pair_freq_score'] +
        w_pair_inv_var * df_merged['pair_inv_var_score'] +
        w_pair_dist * df_merged['pair_mean_dist_score'] # Use log-based closeness
    )

    # --- 4. Filter Based on Score, Respecting Min Samples ---
    return df_merged.sort_values('bias_score', ascending=False)

def filter_object_abs_distance_scored_log(
    df: pd.DataFrame,
    budget: int,
    min_samples_per_pair: int = 3, # Min samples to keep per object pair
    w_pair_freq: float = 1.0,     # Weight for pair frequency component
    w_pair_inv_var: float = 1.0,         # Weight for inverse variance (orig ratio) component
    w_pair_dist: float = 1.0       # Weight for closeness to mean in log-space
) -> list:
    """
    Identify object_abs_distance questions to eliminate based on a bias score,
    using log-transformed distances for closeness calculation.

    The score combines object pair frequency, inverse distance variance (original std/mean),
    and closeness to the pair's mean *log*-distance.

    Args:
        df: DataFrame with object_abs_distance questions.
        budget: Total number of questions to eliminate.
        min_samples_per_pair: Min samples to keep for any object pair.
        w_pair_freq: Weight for normalized pair frequency score component.
        w_pair_inv_var: Weight for normalized inverse variance (low std/mean of original distances).
        w_pair_dist: Weight for normalized closeness to mean in log-space.

    Returns:
        List of question IDs to eliminate.
    """
    df_sorted = get_object_abs_distance_scored_log(df, w_pair_freq, w_pair_inv_var, w_pair_dist)

    remove_ids = []
    kept_counts = df_sorted['object_pair'].value_counts().to_dict()

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        pair_name = row['object_pair']
        if kept_counts.get(pair_name, 0) > min_samples_per_pair:
            remove_ids.append(row['id'])
            kept_counts[pair_name] -= 1

    return remove_ids

def get_object_abs_distance_scored_log_global(
    df: pd.DataFrame,
    w_pair_freq: float = 1.0,
    w_pair_inv_var: float = 1.0,
    w_pair_dist: float = 1.0,
    w_global_dist: float = 1.0 # Weight for global log-normal typicality
) -> pd.DataFrame:
    """
    Identify object_abs_distance questions to eliminate based on a bias score.

    Score combines: object pair frequency, inverse distance variance (orig ratio),
    closeness to pair's mean log-distance, AND typicality based on distance
    from global mean in log-space.

    Args:
        df: DataFrame with object_abs_distance questions.
        w_pair_freq: Weight for pair frequency component.
        w_pair_inv_var: Weight for inverse variance (orig ratio) component.
        w_pair_dist: Weight for closeness to mean in log-space component.
        w_global_dist: Weight for global log-normal PDF score component.


    Returns:
        DataFrame with bias scores and other relevant columns.
    """

    df_filtered = df[df['question_type'] == 'object_abs_distance'].copy()
    if df_filtered.empty:
        return pd.DataFrame(columns=['id'])

    # --- 1. Preprocessing, Calculate Stats (Original and Log) ---
    if 'object_pair' not in df_filtered.columns:
        def extract_objects(question):
            match = re.search(r'between the (.*?) and the (.*?)(?: \(in meters\))?\?$', question)
            if match:
                objs = sorted([match.group(1).strip(), match.group(2).strip()])
                return '_'.join(objs)
            return None
        df_filtered['object_pair'] = df_filtered['question'].apply(extract_objects)
        df_filtered.dropna(subset=['object_pair'], inplace=True)

    if 'ground_truth_num' not in df_filtered.columns:
        df_filtered['ground_truth_num'] = pd.to_numeric(
            df_filtered['ground_truth'], errors='coerce'
        )
        df_filtered.dropna(subset=['ground_truth_num'], inplace=True)

    # Add epsilon for log transformation and prevent issues with zero distance
    epsilon = 1e-6
    valid_distances = df_filtered['ground_truth_num'][df_filtered['ground_truth_num'] > 0]
    if valid_distances.empty:
        print("Warning: No positive distance values found.")
        raise ValueError("No positive distance values found.")
    if len(valid_distances) != len(df_filtered['ground_truth_num']):
        raise ValueError("Some distances are zero or negative, which is not allowed. Please check your data.")

    # shift by 1 so that all distances are positive
    df_filtered['log_ground_truth_num'] = np.log10(df_filtered['ground_truth_num'] + 1.0)

    # Group by object pair and compute stats
    pair_stats = df_filtered.groupby('object_pair').agg(
        count=('id', 'count'),
        mean_log=('log_ground_truth_num', 'mean'),
        std_log=('log_ground_truth_num', 'std')
    ).reset_index()

    pair_stats['std_log'] = pair_stats['std_log'].fillna(0)
    pair_stats['ratio_log'] = (pair_stats['std_log'] / (pair_stats['mean_log'] + epsilon)).fillna(0)

    # --- 2. Global Distance Calculation ---
    df_merged = pd.merge(df_filtered, pair_stats, on='object_pair', how='left')

    # Calculate distance from the global mean in log-space
    global_mean_log = df_merged['log_ground_truth_num'].mean()
    global_std_log = df_merged['log_ground_truth_num'].std()

    # --- 3. Calculate Bias Score Components ---
    # Pair Frequency Component
    df_merged['pair_freq_score'] = minmax_scale(df_merged['count'])

    # Inverse Variance Component (using original std/mean ratio)
    df_merged['pair_inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio_log'] + epsilon)

    # Closeness to Pair Mean Component (in log-space)
    df_merged['norm_dist_from_pair_mean_log'] = (
        abs(df_merged['log_ground_truth_num'] - df_merged['mean_log']) /
        (df_merged['std_log'] + epsilon)
    )
    df_merged['pair_mean_dist_score'] = 1.0 - minmax_scale(df_merged['norm_dist_from_pair_mean_log'] + epsilon)

    # Global Typicality Component (using distance from global mean in log-space)
    df_merged['global_dist_from_mean_log'] = abs(df_merged['log_ground_truth_num'] - global_mean_log) / (global_std_log + epsilon)

    # Normalize the global distance from mean
    df_merged['global_mean_dist_score'] = 1.0 - minmax_scale(df_merged['global_dist_from_mean_log'] + epsilon)


    # --- 4. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_pair_freq * df_merged['pair_freq_score'] +
        w_pair_inv_var * df_merged['pair_inv_var_score'] +
        w_pair_dist * df_merged['pair_mean_dist_score'] +
        w_global_dist * df_merged['global_mean_dist_score']
    )

    # --- 5. Filter Based on Score, Respecting Min Samples ---
    df_sorted = df_merged.sort_values('bias_score', ascending=False)
    return df_sorted

def filter_object_abs_distance_scored_log_global(
    df: pd.DataFrame,
    budget: int,
    min_samples_per_pair: int = 0,  # Min samples to keep per object pair
    w_pair_freq: float = 1.0,
    w_pair_inv_var: float = 1.0,
    w_pair_dist: float = 1.0,
    w_global_dist: float = 1.0  # Weight for global log-normal typicality
) -> list:
    """
    Identify object_abs_distance questions to eliminate based on a bias score.

    Score combines: object pair frequency, inverse distance variance (orig ratio),
    closeness to pair's mean log-distance, AND typicality based on global
    log-normal fit of all distances for this question type.

    Args:
        df: DataFrame with object_abs_distance questions.
        budget: Total number of questions to eliminate.
        min_samples_per_pair: Min samples to keep for any object pair.
        w_pair_freq: Weight for pair frequency component.
        w_pair_inv_var: Weight for inverse variance (orig ratio) component.
        w_pair_dist: Weight for closeness to mean in log-space component.
        w_global_dist: Weight for global log-normal PDF score component.


    Returns:
        List of question IDs to eliminate.
    """
    df_sorted = get_object_abs_distance_scored_log_global(
        df, w_pair_freq, w_pair_inv_var, w_pair_dist, w_global_dist
    )

    remove_ids = []
    kept_counts = df_sorted['object_pair'].value_counts().to_dict()

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        pair_name = row['object_pair']
        if kept_counts.get(pair_name, 0) > min_samples_per_pair:
            remove_ids.append(row['id'])
            kept_counts[pair_name] -= 1

    return remove_ids



from scipy.stats import lognorm # Import lognorm from scipy


def get_object_abs_distance_scored_log_global_pdf(
    df: pd.DataFrame,
    w_pair_freq: float = 1.0,
    w_pair_inv_var: float = 1.0,
    w_pair_dist: float = 1.0,
    w_global_dist: float = 1.0 # Weight for global log-normal typicality
) -> pd.DataFrame:
    """
    Identify object_abs_distance questions to eliminate based on a bias score.

    Score combines: object pair frequency, inverse distance variance (orig ratio),
    closeness to pair's mean log-distance, AND typicality based on global
    log-normal fit of all distances for this question type.

    Args:
        df: DataFrame with object_abs_distance questions.
        w_pair_freq: Weight for pair frequency component.
        w_pair_inv_var: Weight for inverse variance (orig ratio) component.
        w_pair_dist: Weight for closeness to mean in log-space component.
        w_global_dist: Weight for global log-normal PDF score component.


    Returns:
        DataFrame with bias scores and other relevant columns.
    """

    df_filtered = df[df['question_type'] == 'object_abs_distance'].copy()
    if df_filtered.empty:
        return pd.DataFrame(columns=['id'])

    # --- 1. Preprocessing, Calculate Stats (Original and Log) ---
    if 'object_pair' not in df_filtered.columns:
        def extract_objects(question):
            match = re.search(r'between the (.*?) and the (.*?)(?: \(in meters\))?\?$', question)
            if match:
                objs = sorted([match.group(1).strip(), match.group(2).strip()])
                return '_'.join(objs)
            return None
        df_filtered['object_pair'] = df_filtered['question'].apply(extract_objects)
        df_filtered.dropna(subset=['object_pair'], inplace=True)

    if 'ground_truth_num' not in df_filtered.columns:
        df_filtered['ground_truth_num'] = pd.to_numeric(
            df_filtered['ground_truth'], errors='coerce'
        )
        df_filtered.dropna(subset=['ground_truth_num'], inplace=True)

    # Add epsilon for log transformation and prevent issues with zero distance
    epsilon = 1e-6
    valid_distances = df_filtered['ground_truth_num'][df_filtered['ground_truth_num'] > 0]
    if valid_distances.empty:
        print("Warning: No positive distance values found.")
        raise ValueError("No positive distance values found.")
    if len(valid_distances) != len(df_filtered['ground_truth_num']):
        raise ValueError("Some distances are zero or negative, which is not allowed. Please check your data.")

    # shift by 1 so that all distances are positive
    df_filtered['log_ground_truth_num'] = np.log10(df_filtered['ground_truth_num'] + 1.0)

    # Group by object pair and compute stats
    pair_stats = df_filtered.groupby('object_pair').agg(
        count=('id', 'count'),
        mean=('ground_truth_num', 'mean'),
        std=('ground_truth_num', 'std'),
        mean_log=('log_ground_truth_num', 'mean'),
        std_log=('log_ground_truth_num', 'std')
    ).reset_index()

    pair_stats['std_log'] = pair_stats['std_log'].fillna(0)
    pair_stats['ratio_log'] = (pair_stats['std_log'] / (pair_stats['mean_log'] + epsilon)).fillna(0)

    # --- 2. Fit Global Log-Normal Distribution ---
    try:
        # Fit log-normal to all positive distances in this subset
        shape, loc, scale = lognorm.fit(valid_distances, floc=0) # Force location to 0
        print(f"Global Lognormal fit: shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f}")
        global_dist_fitted = True
    except Exception as e:
        print(f"Warning: Could not fit global log-normal distribution: {e}")
        global_dist_fitted = False

    # --- 3. Calculate Bias Score Components ---
    df_merged = pd.merge(df_filtered, pair_stats, on='object_pair', how='left')

    # Pair Frequency Component
    df_merged['pair_freq_score'] = minmax_scale(df_merged['count'])

    # Inverse Variance Component (using original std/mean ratio)
    # df_merged['pair_inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio_orig'] + epsilon)
    df_merged['pair_inv_var_score'] = 1.0 - minmax_scale(df_merged['ratio_log'] + epsilon)

    # Closeness to Pair Mean Component (in log-space)
    df_merged['norm_dist_from_pair_mean_log'] = (
        abs(df_merged['log_ground_truth_num'] - df_merged['mean_log']) /
        (df_merged['std_log'] + epsilon)
    )
    df_merged['pair_mean_dist_score'] = 1.0 - minmax_scale(df_merged['norm_dist_from_pair_mean_log'] + epsilon)

    # Global Typicality Component (using log-normal PDF)
    if global_dist_fitted:
        # Calculate PDF for each sample's distance under the global fit
        df_merged['global_pdf'] = lognorm.pdf(df_merged['ground_truth_num'], shape, loc=loc, scale=scale)
        # Normalize PDF values to get score (higher PDF = more typical = higher score)
        df_merged['global_typicality_score'] = minmax_scale(df_merged['global_pdf'])
    else:
        # If fit failed, assign a neutral score (or zero)
        df_merged['global_typicality_score'] = 0.0


    # --- 4. Calculate Final Bias Score ---
    df_merged['bias_score'] = (
        w_pair_freq * df_merged['pair_freq_score'] +
        w_pair_inv_var * df_merged['pair_inv_var_score'] +
        w_pair_dist * df_merged['pair_mean_dist_score'] +
        w_global_dist * df_merged['global_typicality_score']
    )

    # --- 5. Filter Based on Score, Respecting Min Samples ---
    df_sorted = df_merged.sort_values('bias_score', ascending=False)
    return df_sorted


def filter_object_abs_distance_scored_log_global_pdf(
    df: pd.DataFrame,
    budget: int,
    min_samples_per_pair: int = 0,  # Min samples to keep per object pair
    w_pair_freq: float = 1.0,
    w_pair_inv_var: float = 1.0,
    w_pair_dist: float = 1.0,
    w_global_dist: float = 1.0  # Weight for global log-normal typicality
) -> list:
    """
    Identify object_abs_distance questions to eliminate based on a bias score.

    Score combines: object pair frequency, inverse distance variance (orig ratio),
    closeness to pair's mean log-distance, AND typicality based on global
    log-normal fit of all distances for this question type.

    Args:
        df: DataFrame with object_abs_distance questions.
        budget: Total number of questions to eliminate.
        min_samples_per_pair: Min samples to keep for any object pair.
        w_pair_freq: Weight for pair frequency component.
        w_pair_inv_var: Weight for inverse variance (orig ratio) component.
        w_pair_dist: Weight for closeness to mean in log-space component.
        w_global_dist: Weight for global log-normal PDF score component.


    Returns:
        List of question IDs to eliminate.
    """
    df_sorted = get_object_abs_distance_scored_log_global_pdf(
        df, w_pair_freq, w_pair_inv_var, w_pair_dist, w_global_dist
    )

    remove_ids = []
    kept_counts = df_sorted['object_pair'].value_counts().to_dict()

    for _, row in df_sorted.iterrows():
        if len(remove_ids) >= budget:
            break # Budget met

        pair_name = row['object_pair']
        if kept_counts.get(pair_name, 0) > min_samples_per_pair:
            remove_ids.append(row['id'])
            kept_counts[pair_name] -= 1

    return remove_ids


def filter_room_size_estimation_v1(df, budget, logscale=False):
    """
    Identify room_size_estimation questions to eliminate.

    Args:
        df: DataFrame with room_size_estimation questions
        budget: Number of questions to eliminate
        logscale: Whether to use log scale for ground truth values

    Returns:
        List of question IDs to eliminate
    """
    qdf = df[df['question_type'] == 'room_size_estimation'].copy()

    # Convert ground_truth to numeric
    qdf['ground_truth_num'] = pd.to_numeric(qdf['ground_truth'], errors='coerce')
    qdf['log_ground_truth_num'] = np.log(qdf['ground_truth_num'] + 1e-6)

    # Calculate mean and standard deviation
    mean_size = qdf['ground_truth_num'].mean()
    log_mean_size = qdf['log_ground_truth_num'].mean()

    # Sort by closeness to the mean
    qdf['distance_to_mean'] = abs(qdf['ground_truth_num'] - mean_size)
    qdf['log_distance_to_mean'] = abs(qdf['log_ground_truth_num'] - log_mean_size)

    qdf_sorted = qdf.sort_values('log_distance_to_mean' if logscale else 'distance_to_mean')

    # Select the specified number of questions closest to the mean
    remove_ids = qdf_sorted.head(budget)['id'].tolist()

    return remove_ids[:budget]  # Ensure we don't exceed budget

def get_room_size_estimation_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates bias scores for room_size_estimation questions.
    The score reflects closeness to the global mean size in log-space.
    Higher score means closer to the mean (more typical/biased).

    Args:
        df: DataFrame containing room_size_estimation questions.
            Requires 'id', 'question_type', 'ground_truth'.

    Returns:
        DataFrame with 'id' and 'bias_score' columns.
    """
    qdf = df[df['question_type'] == 'room_size_estimation'].copy()
    if qdf.empty:
        return pd.DataFrame(columns=['id', 'bias_score'])

    # --- 1. Preprocessing ---
    if 'ground_truth_num' not in qdf.columns:
        qdf['ground_truth_num'] = pd.to_numeric(qdf['ground_truth'], errors='coerce')
        qdf.dropna(subset=['ground_truth_num'], inplace=True)

    if qdf.empty: return pd.DataFrame(columns=['id', 'bias_score'])

    epsilon = 1e-9 # Use a small epsilon for log and division stability
    qdf['log_ground_truth_num'] = np.log(qdf['ground_truth_num'] + epsilon)

    # --- 2. Calculate Distance from Global Mean (Log-Space) ---
    log_mean_size = qdf['log_ground_truth_num'].mean()
    qdf['log_distance_to_mean'] = abs(qdf['log_ground_truth_num'] - log_mean_size)

    # --- 3. Calculate Bias Score ---
    # Score = 1 - normalized distance. Higher score = closer to mean.
    # Add epsilon before scaling in case all distances are identical
    qdf['bias_score'] = 1.0 - minmax_scale(qdf['log_distance_to_mean'] + epsilon)
    # qdf['bias_score'] = 1.0 / (minmax_scale(qdf['log_distance_to_mean']) + epsilon)

    return qdf


def filter_room_size_estimation_sampled(
    df: pd.DataFrame,
    budget: int,
    alpha: float = 1.0, # Scale for bias score
    seed: int = 42,
) -> list:
    """
    Filters room_size_estimation questions via weighted sampling based on bias score.
    Samples are more likely to be removed if their size is close to the global
    mean size (in log-space), as indicated by a higher bias score.

    Args:
        df: DataFrame with room_size_estimation questions.
        budget: The exact number of questions to remove.
        alpha: Power factor to increase concentration around the mean (>= 1.0).
            Higher values sample more aggressively near the mean.
        seed: Random seed for sampling reproducibility.

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_room_size_estimation_score(df)
    if df_with_scores.empty:
        print("Warning: No room size estimation questions found.")
        return []

    # Ensure budget is not larger than available samples
    budget = min(budget, len(df_with_scores))
    if budget <= 0:
        return []

    # Apply power scaling to the bias score to concentrate weights near the mean
    # Add epsilon before scaling in case bias_score is exactly 0
    epsilon_weight = 1e-9
    # Ensure scale is at least 1
    alpha = max(1.0, alpha)
    weights = (df_with_scores['bias_score'] + epsilon_weight) ** alpha

    # Perform weighted sampling *without replacement*
    samples_to_remove = df_with_scores.sample(
        n=budget,
        weights=weights,
        random_state=seed,
        replace=False # Ensure we don't pick the same ID twice
    )
    remove_ids = samples_to_remove['id'].tolist()

    return remove_ids[:budget] # Ensure exact budget

def get_room_size_estimation_score_pdf(
    df: pd.DataFrame,
    alpha: float = 1.0, # Scale for bias score
) -> pd.DataFrame:
    """
    Calculates bias scores for room_size_estimation questions.
    The score reflects closeness to the global mean size in log-space.
    Higher score means closer to the mean (more typical/biased).

    Args:
        df: DataFrame containing room_size_estimation questions.
            Requires 'id', 'question_type', 'ground_truth'.

    Returns:
        DataFrame with 'id' and 'bias_score' columns.
    """
    qdf = df[df['question_type'] == 'room_size_estimation'].copy()
    qdf['ground_truth_num'] = pd.to_numeric(qdf['ground_truth'], errors='coerce')
    qdf.dropna(subset=['ground_truth_num'], inplace=True)

    # Fit a lognormal distribution to the data
    x = qdf['ground_truth_num']
    shape, loc, scale = lognorm.fit(x, floc=0)  # Fix location to 0 for lognormal

    pdf = lognorm.pdf(x, shape, loc, scale)

    weights = pdf ** alpha
    weights /= weights.sum()  # Normalize weights
    qdf['bias_score'] = weights

    return qdf


def filter_room_size_estimation_sampled_pdf(
    df: pd.DataFrame,
    budget: int,
    alpha: float = 1.0,
    seed: int = 42,
) -> list:
    """
    Filters room_size_estimation questions via weighted sampling based on bias score.
    Samples are more likely to be removed if their size is close to the global
    mean size (in log-space), as indicated by a higher bias score.

    Args:
        df: DataFrame with room_size_estimation questions.
        budget: The exact number of questions to remove.
        alpha: Power factor to increase concentration around the mean.
            Higher values sample more aggressively near the mean.
        seed: Random seed for sampling reproducibility.

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_room_size_estimation_score_pdf(df, alpha)
    if df_with_scores.empty:
        print("Warning: No room size estimation questions found.")
        return []

    # Ensure budget is not larger than available samples
    budget = min(budget, len(df_with_scores))
    if budget <= 0:
        return []

    # Perform weighted sampling *without replacement*
    samples_to_remove = df_with_scores.sample(
        n=budget,
        weights=df_with_scores['bias_score'],
        random_state=seed,
        replace=False # Ensure we don't pick the same ID twice
    )
    remove_ids = samples_to_remove['id'].tolist()

    return remove_ids[:budget] # Ensure exact budget

def filter_object_rel_direction_v1(df, budget):
    """
    Identify object_rel_direction_* questions to eliminate.

    Args:
        df: DataFrame with object_rel_direction questions
        budget: Number of questions to eliminate

    Returns:
        List of question IDs to eliminate
    """
    # Count frequency of each answer
    answer_counts = df['ground_truth'].value_counts()

    # Calculate target distribution (ideally uniform)
    total_questions = len(df)
    num_answers = len(answer_counts)
    target_per_answer = total_questions / num_answers

    # Calculate excess per answer (how many more than ideal we have)
    answer_excess = {}
    for answer, count in answer_counts.items():
        excess = max(0, count - target_per_answer)
        answer_excess[answer] = int(excess)

    # Calculate how many to remove from each answer
    total_excess = sum(answer_excess.values())
    remove_per_answer = {}

    if total_excess > 0:
        for answer, excess in answer_excess.items():
            # Proportionally allocate budget based on excess
            remove_per_answer[answer] = min(
                excess,
                int(budget * excess / total_excess) + 1
            )

    # Initialize list of IDs to remove
    remove_ids = []

    # Remove questions from overrepresented answers
    for answer, to_remove in remove_per_answer.items():
        if to_remove <= 0:
            continue

        # Get questions with this answer
        answer_questions = df[df['ground_truth'] == answer]

        # Select random questions to remove (could be made more sophisticated)
        questions_to_remove = answer_questions.sample(min(to_remove, len(answer_questions)))
        remove_ids.extend(questions_to_remove['id'].tolist())

    # If we haven't reached our budget, remove more from the most overrepresented answers
    if len(remove_ids) < budget:
        remaining = budget - len(remove_ids)
        most_common_answers = answer_counts.index.tolist()

        for answer in most_common_answers:
            if len(remove_ids) >= budget:
                break

            # Get questions with this answer that haven't been removed yet
            answer_questions = df[
                (df['ground_truth'] == answer) &
                (~df['id'].isin(remove_ids))
            ]

            # Select random questions to remove
            to_remove = min(remaining, len(answer_questions))
            if to_remove > 0:
                questions_to_remove = answer_questions.sample(to_remove)
                remove_ids.extend(questions_to_remove['id'].tolist())
                remaining -= to_remove

    return remove_ids[:budget]  # Ensure we don't exceed budget

def get_rel_direction_removal_candidates(df_subtype: pd.DataFrame, seed: int = 42, sampled: bool = False) -> dict:
    """
    Identifies potential candidates for removal to balance the GT answer
    distribution within a single object_rel_direction subtype (e.g., easy).

    Args:
        df_subtype: DataFrame filtered for a single question type
                    (e.g., df[df['question_type'] == 'object_rel_direction_easy']).
                    Requires 'id' and 'gt_option' columns.

    Returns:
        A dictionary mapping each ground truth answer option (e.g., 'left')
        to a list of sample IDs that are candidates for removal (the excess samples).
        Returns an empty dict if balancing is not possible or needed.
    """
    gt_counts = df_subtype['gt_option'].value_counts()

    if len(gt_counts) <= 1:
        return {} # Cannot balance if only one answer type exists

    min_count = gt_counts.min()
    # print(f"Subtype: {df_subtype['question_type'].iloc[0]}, Min count: {min_count}, Counts: {gt_counts.to_dict()}")

    removal_candidates = {}
    for gt_answer, count in gt_counts.items():
        # num_to_remove = count - min_count - 1 # -1 : ok to have one more than min_count
        num_to_remove = count - min_count
        if num_to_remove > 0:
            # Get IDs for this specific ground truth answer
            ids_for_gt = df_subtype[df_subtype['gt_option'] == gt_answer]['id'].tolist()
            # Store these IDs as potential candidates for removal
            # No random sampling here yet, just identify all excess IDs
            if sampled:
                rng = np.random.default_rng(seed)
                rng.shuffle(ids_for_gt)
            removal_candidates[gt_answer] = ids_for_gt[:num_to_remove] # Take the first N excess IDs

    return removal_candidates

def filter_object_rel_direction_v2(
    df: pd.DataFrame,
    budget: int, # Budget might be partially ignored if full balancing requires more removals
    seed: int = 42, # Seed for reproducibility
    shuffled: bool = True, # Whether to shuffle the IDs for random sampling
    verbose: bool = False
) -> list:
    """
    Filters object_rel_direction questions by balancing the ground truth answer
    distribution within each subtype (easy, medium, hard).

    It identifies excess samples for frequent answers within each subtype and
    randomly selects samples for removal to match the least frequent answer count.

    Args:
        df: DataFrame containing all object_rel_direction questions.
            Requires 'id', 'question_type', 'options', 'ground_truth'.
        budget: The maximum *target* number of questions to remove. The actual number
                removed might be lower if full balancing requires fewer removals,
                or potentially higher if strict balancing is prioritized over budget.
                (Current implementation prioritizes balancing).
        seed: Random seed for sampling.


    Returns:
        List of question IDs selected for elimination.
    """
    random.seed(seed)
    subtypes = [
        'object_rel_direction_easy',
        'object_rel_direction_medium',
        'object_rel_direction_hard'
    ]

    # --- Identify all candidates needed for balancing per subtype ---
    all_candidates_by_subtype = {} # Store {subtype: {gt_answer: [ids]}}
    total_candidates_for_balance = 0

    for subtype in subtypes:
        df_subtype = df[df['question_type'] == subtype].copy()
        if df_subtype.empty:
            continue

        # Extract GT answer string if not present
        # Assuming 'options' and 'ground_truth' (A/B/C/D) exist
        df_subtype["gt_idx"] = df_subtype["ground_truth"].apply(lambda x: "ABCD".index(x))
        df_subtype["gt_option"] = df_subtype.apply(lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1)

        # Extract GT option if needed (will be done by get_rel_direction_removal_candidates)
        subtype_candidates = get_rel_direction_removal_candidates(df_subtype, seed=seed, sampled=shuffled)
        all_candidates_by_subtype[subtype] = subtype_candidates
        for gt_answer, ids in subtype_candidates.items():
            total_candidates_for_balance += len(ids)

    if verbose:
        print(f"Total samples identified for full balancing: {total_candidates_for_balance}")

    # --- Select samples for removal using random sampling ---
    # Current approach: Perform full balancing for each subtype independently.
    # The budget parameter acts more like a suggestion here; we prioritize balance.
    final_remove_ids = set()
    for subtype, candidates in all_candidates_by_subtype.items():
        for gt_answer, ids_to_sample_from in candidates.items():
            # These are already the excess IDs needed for balancing
            # No further sampling needed based on count, just add them.
            # If random selection *among* the excess was desired, sample here.
            # For now, remove all identified excess to achieve balance.
            final_remove_ids.update(ids_to_sample_from)

    if verbose:
        print(f"Actual number of samples selected for removal to balance: {len(final_remove_ids)}")

    # If we strictly needed to adhere to budget (and budget < total_candidates):
    # Option: Randomly sample 'budget' IDs from 'final_remove_ids'
    if len(final_remove_ids) > budget:
        if verbose:
            print(f"Warning: Balancing requires {len(final_remove_ids)} removals, exceeding budget {budget}. Selecting subset.")
        final_remove_ids_list = list(final_remove_ids)
        random.shuffle(final_remove_ids_list)
        final_remove_ids = set(final_remove_ids_list[:budget])

    return list(final_remove_ids)

def get_object_rel_direction_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a bias score for object_rel_direction questions based on the
    frequency of the objects involved in the question prompt.

    Args:
        df: DataFrame containing object_rel_direction questions.
            Requires 'id', 'question_type', 'question', 'options', 'ground_truth'.

    Returns:
        DataFrame with 'id', 'question_type', 'gt_option', and 'bias_score'
        for relevant questions.
    """
    qdf = df[df["question_type"].str.startswith("object_rel_direction")].copy()
    if qdf.empty:
        return pd.DataFrame(columns=['id', 'question_type', 'gt_option', 'bias_score'])

    # --- 1. Extract Objects and GT Option ---

    # Extract objects from question
    qdf[["positioning_object", "orienting_object", "querying_object"]] = \
        qdf["question"].str.extract(r'standing by the (.*?) and facing the (.*?), is the (.*?) to')
    # Clean object names
    for col in ["positioning_object", "orienting_object", "querying_object"]:
        qdf[col] = qdf[col].str.strip()

    # Extract GT answer string
    qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
    qdf["gt_option"] = qdf.apply(lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1)

    # Drop rows where extraction failed
    qdf.dropna(subset=["positioning_object", "orienting_object", "querying_object", "gt_option"], inplace=True)
    if qdf.empty:
        return pd.DataFrame(columns=['id', 'question_type', 'gt_option', 'bias_score'])


    # --- 2. Calculate Overall Object Frequencies ---
    all_objects = pd.concat([
        qdf["positioning_object"],
        qdf["orienting_object"],
        qdf["querying_object"]
    ]).dropna().tolist()
    obj_freq = Counter(all_objects)

    if not obj_freq:
        print("Warning: Could not calculate object frequencies.")
        qdf['bias_score'] = 0.0
        return qdf[['id', 'question_type', 'gt_option', 'bias_score']]

    # --- 3. Normalize Frequencies ---
    obj_freq_values = np.array(list(obj_freq.values())).reshape(-1, 1)
    scaled_obj_freq_values = minmax_scale(obj_freq_values) if len(np.unique(obj_freq_values)) > 1 else np.ones_like(obj_freq_values)
    norm_obj_freq_map = {key: scaled_obj_freq_values[i][0] for i, key in enumerate(obj_freq.keys())}

    # --- 4. Calculate Bias Score per Sample ---
    def calculate_score(row):
        score = (
            norm_obj_freq_map.get(row["positioning_object"], 0) +
            norm_obj_freq_map.get(row["orienting_object"], 0) +
            norm_obj_freq_map.get(row["querying_object"], 0)
        )
        return score

    qdf['bias_score'] = qdf.apply(calculate_score, axis=1)

    return qdf

def filter_object_rel_direction_scored(
    df: pd.DataFrame, # Should be the output of get_object_rel_direction_score
    budget: int, # Still potentially ignored if balancing requires more/less
    seed: int = 42 # Seed for reproducibility
) -> list:
    """
    Filters object_rel_direction questions by balancing the ground truth answer
    distribution within each subtype, prioritizing removal of higher-scoring samples.

    Args:
        df_with_scores: DataFrame containing 'id', 'question_type', 'gt_option',
                        and 'bias_score' for object_rel_direction questions.
        budget: The maximum target number of questions to remove.
                (Current implementation prioritizes balancing).

    Returns:
        List of question IDs selected for elimination.
    """
    random.seed(seed)

    df_with_scores = get_object_rel_direction_score(df)

    if df_with_scores.empty or not all(col in df_with_scores.columns for col in ['id', 'question_type', 'gt_option', 'bias_score']):
        print("Warning: Input DataFrame missing required columns.")
        return []

    subtypes = [
        'object_rel_direction_easy',
        'object_rel_direction_medium',
        'object_rel_direction_hard'
    ]
    final_remove_ids = set()
    total_candidates_for_balance = 0 # Just for reporting

    for subtype in subtypes:
        df_subtype = df_with_scores[df_with_scores['question_type'] == subtype].copy()
        if df_subtype.empty:
            continue

        gt_counts = df_subtype['gt_option'].value_counts()
        if len(gt_counts) <= 1:
            continue # Cannot balance

        min_count = gt_counts.min()
        # print(f"Subtype: {subtype}, Min count: {min_count}, Counts: {gt_counts.to_dict()}")

        for gt_answer, count in gt_counts.items():
            num_to_remove = count - min_count
            if num_to_remove > 0:
                total_candidates_for_balance += num_to_remove
                # Get samples for this specific ground truth answer
                gt_samples = df_subtype[df_subtype['gt_option'] == gt_answer]
                # Sort them by bias_score (highest first)
                gt_samples_sorted = gt_samples.sort_values('bias_score', ascending=False)
                # Select the top 'num_to_remove' IDs based on score
                ids_to_remove_for_gt = gt_samples_sorted['id'].head(num_to_remove).tolist()
                final_remove_ids.update(ids_to_remove_for_gt)

    print(f"Total samples identified for full balancing: {total_candidates_for_balance}")
    print(f"Actual number of samples selected for removal (prioritizing high bias score): {len(final_remove_ids)}")

    # Optional: Enforce budget strictly (as before)
    if len(final_remove_ids) > budget:
        print(f"Warning: Balancing requires {len(final_remove_ids)} removals, exceeding budget {budget}. Selecting subset.")
        final_remove_ids_list = list(final_remove_ids)
        random.shuffle(final_remove_ids_list) # Maybe sort by score instead?
        final_remove_ids = set(final_remove_ids_list[:budget])

    return list(final_remove_ids)

def get_object_rel_distance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a bias score for object_rel_distance questions.
    The score is the maximum probability an option is correct based on
    global frequencies of (closer_object, target_object) pairs.

    Args:
        df: DataFrame containing at least the object_rel_distance questions.
            Requires columns: 'id', 'question_type', 'question', 'options', 'ground_truth'.

    Returns:
        DataFrame with added 'bias_score' column for relevant questions.
    """
    df_filtered = df[df['question_type'] == 'object_rel_distance'].copy()
    if df_filtered.empty:
        return pd.DataFrame(columns=['id', 'bias_score'])

    # --- 1. Extract Target and GT Object Name ---
    # Extract target object (last object mentioned in question)
    df_filtered['target_object'] = df_filtered['question'].str.extract(r'is the closest to the (.*?)\?$')[0].str.strip()

    # Extract GT object name from options
    try:
        df_filtered["gt_idx"] = df_filtered["ground_truth"].apply(lambda x: "ABCD".index(x))
        if isinstance(df_filtered["options"].iloc[0], str):
            df_filtered["options_list"] = df_filtered["options"].apply(
                lambda x: [opt.split('. ', 1)[1] for opt in x.split("'")[1::2]]
            )
        else:
            df_filtered["options_list"] = df_filtered["options"]

        # Get the string for the ground truth option
        df_filtered["gt_option_str"] = df_filtered.apply(
            lambda row: row["options_list"][row["gt_idx"]], axis=1
        )
        # Ensure no leading/trailing spaces in gt object name
        df_filtered["gt_object_name"] = df_filtered["gt_option_str"].str.strip()

    except Exception as e:
        print(f"Error parsing options/GT for rel_distance: {e}")
        df_filtered['bias_score'] = 0.0 # Assign neutral score if parsing fails
        return df_filtered[['id', 'bias_score']]

    # --- 2. Calculate Global Frequencies of Correct Pairs ---
    # Pair = (target_object, gt_object_name), sorted alphabetically
    global_pair_counts = Counter()
    for _, row in df_filtered.iterrows():
        # Ensure both objects are valid strings before creating pair
        if isinstance(row['target_object'], str) and isinstance(row['gt_object_name'], str):
            pair = tuple(sorted([row['target_object'], row['gt_object_name']]))
            global_pair_counts[pair] += 1

    print(f"Global pair counts: {global_pair_counts}")

    # --- 3. Calculate Max Option Probability (Bias Score) for Each Question ---
    bias_scores = []
    for index, row in df_filtered.iterrows():
        target_obj = row['target_object']
        option_strings = row['options_list']
        option_objects = [opt.strip() for opt in option_strings]

        option_pair_freqs = []
        valid_options = True
        if not isinstance(target_obj, str): # Skip if target object is invalid
            valid_options = False
        else:
            for opt_obj in option_objects:
                if not isinstance(opt_obj, str): # Skip if option object is invalid
                    valid_options = False
                    break
                # Form pair and get its global frequency
                pair = tuple(sorted([target_obj, opt_obj]))
                option_pair_freqs.append(global_pair_counts.get(pair, 0))

        if not valid_options or not option_pair_freqs:
            max_prob = 0.0 # Assign neutral score if objects are invalid
        else:
            total_freq_sum = sum(option_pair_freqs)
            if total_freq_sum > 0:
                # Calculate probability for each option
                probabilities = [freq / total_freq_sum for freq in option_pair_freqs]
                max_prob = max(probabilities)
            else:
                # If no pairs found in global stats, prob is undefined/neutral
                max_prob = 0.0 # Or perhaps 1/num_options? Let's use 0 for bias score.

        bias_scores.append(max_prob)

    df_filtered['bias_score'] = bias_scores

    # Select and return relevant columns
    return df_filtered

def filter_object_rel_distance_score(
    df: pd.DataFrame,
    budget: int,
) -> list:
    """
    Filters object_rel_distance questions based on the pre-calculated bias score
    (max option probability). Removes questions with the highest scores.

    Args:
        df: DataFrame containing 'id' and 'bias_score' columns for
            object_rel_distance questions (output of get_object_rel_distance_score).
        budget: The maximum number of questions to remove.

    Returns:
        List of question IDs to eliminate.
    """

    df_filtered = get_object_rel_distance_score(df)
    if df_filtered.empty:
        return []

    # Sort by score (highest probability first)
    df_sorted = df_filtered.sort_values('bias_score', ascending=False)

    # Remove top N based on budget
    # Note: No min_samples constraint applied here, as the score is probability based.
    # We could add a check for min_samples_per_target_object if needed.
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids

def get_object_rel_distance_score_v2(
        df: pd.DataFrame,
        w_gt_obj: float = 1.0,
        w_pair_freq: float = 1.0,
        w_ord_pair_freq: float = 1.0,
    ) -> pd.DataFrame:
    """
    Calculates a bias score for object_rel_distance questions.
    The score is the maximum probability an option is correct based on
    global frequencies of (closer_object, target_object) pairs.

    Args:
        df: DataFrame containing at least the object_rel_distance questions.
            Requires columns: 'id', 'question_type', 'question', 'options', 'ground_truth'.
        w_gt_obj: Weight for the ground truth object name component.
        w_pair_freq: Weight for the pair frequency component.
        w_ord_pair_freq: Weight for the ordered pair frequency component.

    Returns:
        DataFrame with added 'bias_score' column for relevant questions.
    """
    qdf = df[df['question_type'] == 'object_rel_distance'].copy()

    # --- 1. Extract Target and GT Object Name ---
    # Extract target object (last object mentioned in question)
    qdf[["object_1", "object_2", "object_3", "object_4", "target_object"]] = qdf["question"].str.extract(r'which of these objects \((.*?), (.*?), (.*?), (.*?)\) is the closest to the (.*?)\?$')

    # Extract GT object name from options
    qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
    qdf["gt_option"] = qdf.apply(lambda row: row["options"][row["gt_idx"]], axis=1)
    # Ensure no leading/trailing spaces in gt object name
    qdf["gt_object"] = qdf["gt_option"].apply(lambda x: x.split(". ")[-1].strip())

    qdf["tgt_gt_pair"] = qdf.apply(lambda row: "-".join(sorted([row["target_object"], row["gt_object"]])), axis=1)
    qdf["tgt_gt_ord_pair"] = qdf.apply(lambda row: "-".join([row["target_object"], row["gt_object"]]), axis=1)

    # --- 2. Calculate Global Frequencies of Correct Pairs ---
    def get_counts(col):
        return qdf.groupby(col).size().reset_index(name='count').set_index(col)["count"].to_dict()

    gt_obj_counts = get_counts("gt_object")
    global_pair_counts = get_counts("tgt_gt_pair")
    global_ordered_pair_counts = get_counts("tgt_gt_ord_pair")

    # --- 3. Calculate Max Option Probability (Bias Score) for Each Question ---
    bias_info = []
    for index, row in qdf.iterrows():
        option_objects = row[['object_1', 'object_2', 'object_3', 'object_4']].tolist()
        target_obj = row['target_object']

        gt_obj_freqs = []
        pair_freqs = []
        ord_pair_freqs = []

        for opt_obj in option_objects:
            # Form pair and get its global frequency
            tgt_gt_pair = "-".join(sorted([target_obj, opt_obj]))
            tgt_gt_ord_pair = "-".join([target_obj, opt_obj])

            gt_obj_freqs.append(gt_obj_counts.get(opt_obj, 0))
            pair_freqs.append(global_pair_counts.get(tgt_gt_pair, 0))
            ord_pair_freqs.append(global_ordered_pair_counts.get(tgt_gt_ord_pair, 0))

        def get_max_prob(freqs):
            if len(freqs) == 0:
                return 0.0
            total_freq_sum = sum(freqs)
            if total_freq_sum <= 0:
                return 0.0
            probs = [freq / total_freq_sum for freq in freqs]
            return max(probs)

        max_gt_obj_prob = get_max_prob(gt_obj_freqs)
        max_pair_prob = get_max_prob(pair_freqs)
        max_ord_pair_prob = get_max_prob(ord_pair_freqs)


        bias_info.append({
            'id': row['id'],
            'gt_obj_prob': max_gt_obj_prob,
            'pair_prob': max_pair_prob,
            'ord_pair_prob': max_ord_pair_prob,
        })

    # Create a new DataFrame from the bias_info list
    bias_df = pd.DataFrame(bias_info)
    # Merge the bias scores back into the original DataFrame
    qdf = pd.merge(qdf, bias_df, on='id', how='left')

    qdf['bias_score'] = (
        w_gt_obj * qdf['gt_obj_prob'] +
        w_pair_freq * qdf['pair_prob'] +
        w_ord_pair_freq * qdf['ord_pair_prob']
    )

    # Select and return relevant columns
    return qdf


def filter_object_rel_distance_score_v2(
    df: pd.DataFrame,
    budget: int,
    w_gt_obj: float = 1.0,
    w_pair_freq: float = 1.0,
    w_ord_pair_freq: float = 1.0,
    ) -> list:
    """
    Filters object_rel_distance questions based on the pre-calculated bias score
    (max option probability). Removes questions with the highest scores.

    Args:
        df: DataFrame containing 'id' and 'bias_score' columns for
            object_rel_distance questions (output of get_object_rel_distance_score).
        budget: The maximum number of questions to remove.
        w_gt_obj: Weight for the ground truth object name component.
        w_pair_freq: Weight for the pair frequency component.
        w_ord_pair_freq: Weight for the ordered pair frequency component.

    Returns:
        List of question IDs to eliminate.
    """

    df_filtered = get_object_rel_distance_score_v2(df, w_gt_obj, w_pair_freq, w_ord_pair_freq)
    if df_filtered.empty:
        return []

    # Sort by score (highest probability first)
    df_sorted = df_filtered.sort_values('bias_score', ascending=False)

    # Remove top N based on budget
    # Note: No min_samples constraint applied here, as the score is probability based.
    # We could add a check for min_samples_per_target_object if needed.
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids

def filter_obj_appearance_order_v1(df, budget):
    """
    Identify obj_appearance_order questions to eliminate.

    Args:
        df: DataFrame with obj_appearance_order questions
        budget: Number of questions to eliminate

    Returns:
        List of question IDs to eliminate
    """
    # Count frequency of each answer
    answer_counts = df['ground_truth'].value_counts()

    # Calculate target distribution (ideally uniform)
    total_questions = len(df)
    num_answers = len(answer_counts)
    target_per_answer = total_questions / num_answers

    # Initialize list of IDs to remove
    remove_ids = []

    # Start with most frequent answers
    for answer in answer_counts.index:
        if len(remove_ids) >= budget:
            break

        answer_questions = df[
            (df['ground_truth'] == answer) &
            (~df['id'].isin(remove_ids))
        ]

        # Calculate excess above target
        excess = max(0, len(answer_questions) - target_per_answer)
        to_remove = min(int(excess), budget - len(remove_ids))

        if to_remove > 0:
            questions_to_remove = answer_questions.sample(to_remove, random_state=42)
            remove_ids.extend(questions_to_remove['id'].tolist())

    return remove_ids[:budget]  # Ensure we don't exceed budget


def filter_app_order_position_bias(
    df: pd.DataFrame,
    budget: int,
    # min_samples: Optional constraint can be added later if needed
) -> list:
    """
    Identify obj_appearance_order questions to eliminate based on object-position bias.

    Scores questions based on the frequency of their ground truth objects
    appearing in specific positions (1st, 2nd, 3rd, 4th) within the sequence.

    Args:
        df: DataFrame with obj_appearance_order questions. Must contain
            'id', 'question_type', 'ground_truth', 'options' columns.
        budget: Total number of questions to eliminate.

    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = df[df['question_type'] == 'obj_appearance_order'].copy()
    if df_filtered.empty:
        return []

    # --- 1. Extract Ground Truth Object Sequence ---
    try:
        df_filtered["gt_idx"] = df_filtered["ground_truth"].apply(lambda x: "ABCD".index(x))
        # Ensure options are lists if they are stored as strings
        if isinstance(df_filtered["options"].iloc[0], str):
            # Assuming options are stored like 'A. obj1, obj2...'
            # This parsing might need adjustment based on the exact format
            df_filtered["options_list"] = df_filtered["options"].apply(
                lambda x: [opt.split('. ', 1)[1] for opt in x.split("'")[1::2]] # Simple parse assuming 'A. ...', 'B. ...' format
            )
        else:
            df_filtered["options_list"] = df_filtered["options"]

        df_filtered["gt_option_str"] = df_filtered.apply(
            lambda row: row["options_list"][row["gt_idx"]], axis=1
        )
        # Extract up to 4 objects from the comma-separated string
        for i in range(4):
            df_filtered[f"gt_obj_{i+1}"] = df_filtered["gt_option_str"].apply(
                lambda x: x.split(", ")[i].strip() if len(x.split(", ")) > i else None
            )
    except Exception as e:
        print(f"Error parsing options or ground truth: {e}")
        print("Please ensure 'options' column is a list of strings or a parseable string representation.")
        return []

    # --- 2. Calculate Object-Position Frequencies ---
    position_freq = Counter()
    total_samples = 0
    for i in range(1, 5): # Positions 1 to 4
        col_name = f"gt_obj_{i}"
        # Count non-null objects in each position
        counts = df_filtered[col_name].value_counts()
        for obj, count in counts.items():
            position_freq[(obj, i)] += count
        total_samples += df_filtered[col_name].notna().sum() # Count valid entries

    if not position_freq:
        print("Warning: Could not calculate position frequencies.")
        return []

    # --- 3. Calculate Bias Score for each Sample ---
    df_filtered['bias_score'] = 0.0
    max_possible_freq_sum = 0 # For normalization baseline

    # Calculate raw score (sum of frequencies for the GT objects)
    for i in range(1, 5):
        col_name = f"gt_obj_{i}"
        # Map frequency to each row based on the object in that position
        freq_map = {obj: position_freq.get((obj, i), 0) for obj, _ in position_freq if _ == i}
        # Apply frequencies, default to 0 if object not found or position is None
        current_pos_score = df_filtered[col_name].map(freq_map).fillna(0)
        df_filtered['bias_score'] += current_pos_score
        # Track max possible frequency sum for normalization (optional but good practice)
        if freq_map:
            max_possible_freq_sum += max(freq_map.values())


    # Optional: Normalize score (e.g., divide by number of objects in sequence or by max possible sum)
    # Simple normalization: divide by number of objects present in GT
    num_objects = df_filtered[[f"gt_obj_{i+1}" for i in range(4)]].notna().sum(axis=1)
    df_filtered['bias_score'] = df_filtered['bias_score'] / num_objects.replace(0, 1) # Avoid division by zero


    # --- 4. Filter Based on Score ---
    df_sorted = df_filtered.sort_values('bias_score', ascending=False)

    # Remove top N based on budget (No min_samples constraint for now)
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids

def filter_app_order_scored(
    df: pd.DataFrame,
    budget: int,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_overall: float = 1.0   # Weight for overall object frequency component
    # min_samples: Optional constraint can be added later if needed
) -> list:
    """
    Identify obj_appearance_order questions to eliminate based on a combined bias score.

    Scores questions based on:
    1. Frequency of their ground truth objects appearing in specific positions.
    2. Overall frequency of their ground truth objects within this question type.

    Args:
        df: DataFrame with obj_appearance_order questions. Must contain
            'id', 'question_type', 'ground_truth', 'options' columns.
        budget: Total number of questions to eliminate.
        w_pos: Weight for positional frequency score component.
        w_overall: Weight for overall object frequency score component.


    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = df[df['question_type'] == 'obj_appearance_order'].copy()
    if df_filtered.empty:
        return []

    # --- 1. Extract Ground Truth Object Sequence ---
    try:
        df_filtered["gt_idx"] = df_filtered["ground_truth"].apply(lambda x: "ABCD".index(x))
        if isinstance(df_filtered["options"].iloc[0], str):
            df_filtered["options_list"] = df_filtered["options"].apply(
                lambda x: [opt.split('. ', 1)[1] for opt in x.split("'")[1::2]]
            )
        else:
            df_filtered["options_list"] = df_filtered["options"]

        df_filtered["gt_option_str"] = df_filtered.apply(
            lambda row: row["options_list"][row["gt_idx"]], axis=1
        )
        gt_objects_list = []
        for i in range(4):
            col_name = f"gt_obj_{i+1}"
            df_filtered[col_name] = df_filtered["gt_option_str"].apply(
                lambda x: x.split(", ")[i].strip() if len(x.split(", ")) > i else None
            )
            # Collect all non-null GT objects for overall frequency count
            gt_objects_list.extend(df_filtered[col_name].dropna().tolist())

    except Exception as e:
        print(f"Error parsing options or ground truth: {e}")
        print("Please ensure 'options' column is a list of strings or a parseable string representation.")
        return []

    # --- 2. Calculate Frequencies ---
    # 2a. Object-Position Frequencies
    position_freq = Counter()
    for i in range(1, 5): # Positions 1 to 4
        col_name = f"gt_obj_{i}"
        counts = df_filtered[col_name].value_counts()
        for obj, count in counts.items():
            position_freq[(obj, i)] += count

    # 2b. Overall Object Frequencies (within this question type)
    overall_obj_freq = Counter(gt_objects_list)

    if not position_freq or not overall_obj_freq:
        print("Warning: Could not calculate frequencies.")
        return []

    # --- 3. Normalize Frequencies (Min-Max Scaling to 0-1) ---
    # Extract values for scaling
    pos_freq_values = np.array(list(position_freq.values())).reshape(-1, 1)
    overall_freq_values = np.array(list(overall_obj_freq.values())).reshape(-1, 1)

    # Scale if there's more than one distinct value, otherwise set scale to 1
    scaled_pos_freq_values = minmax_scale(pos_freq_values) if len(np.unique(pos_freq_values)) > 1 else np.ones_like(pos_freq_values)
    scaled_overall_freq_values = minmax_scale(overall_freq_values) if len(np.unique(overall_freq_values)) > 1 else np.ones_like(overall_freq_values)

    # Create lookup maps for normalized frequencies
    norm_pos_freq_map = {key: scaled_pos_freq_values[i][0] for i, key in enumerate(position_freq.keys())}
    norm_overall_freq_map = {key: scaled_overall_freq_values[i][0] for i, key in enumerate(overall_obj_freq.keys())}

    # --- 4. Calculate Combined Bias Score for each Sample ---
    positional_scores = []
    overall_scores = []
    num_objects_list = []

    for _, row in df_filtered.iterrows():
        current_pos_score = 0
        current_overall_score = 0
        num_objects = 0
        for i in range(1, 5):
            obj = row[f"gt_obj_{i}"]
            if pd.notna(obj):
                num_objects += 1
                current_pos_score += norm_pos_freq_map.get((obj, i), 0)
                current_overall_score += norm_overall_freq_map.get(obj, 0)

        positional_scores.append(current_pos_score)
        overall_scores.append(current_overall_score)
        num_objects_list.append(num_objects if num_objects > 0 else 1) # Avoid division by zero

    df_filtered['positional_score_sum'] = positional_scores
    df_filtered['overall_score_sum'] = overall_scores
    df_filtered['num_objects'] = num_objects_list

    # Calculate final weighted score, normalized by number of objects
    df_filtered['bias_score'] = (
        (w_pos * df_filtered['positional_score_sum'] + w_overall * df_filtered['overall_score_sum']) /
        df_filtered['num_objects']
    )


    # --- 5. Filter Based on Score ---
    df_sorted = df_filtered.sort_values('bias_score', ascending=False)

    # Remove top N based on budget
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids

def filter_app_order_scored_v2(
    df: pd.DataFrame,
    budget: int,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_pair: float = 1.0      # Weight for adjacent pair frequency component
    # min_samples: Optional constraint can be added later if needed
) -> list:
    """
    Identify obj_appearance_order questions to eliminate based on a combined bias score.

    Scores questions based on:
    1. Frequency of their ground truth objects appearing in specific positions.
    2. Frequency of adjacent object pairs within the ground truth sequence.

    Args:
        df: DataFrame with obj_appearance_order questions. Must contain
            'id', 'question_type', 'ground_truth', 'options' columns.
        budget: Total number of questions to eliminate.
        w_pos: Weight for positional frequency score component.
        w_pair: Weight for adjacent pair frequency score component.

    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = df[df['question_type'] == 'obj_appearance_order'].copy()
    if df_filtered.empty:
        return []

    # --- 1. Extract Ground Truth Object Sequence ---
    # (Same extraction logic as before)
    try:
        df_filtered["gt_idx"] = df_filtered["ground_truth"].apply(lambda x: "ABCD".index(x))
        if isinstance(df_filtered["options"].iloc[0], str):
            df_filtered["options_list"] = df_filtered["options"].apply(
                lambda x: [opt.split('. ', 1)[1] for opt in x.split("'")[1::2]]
            )
        else:
            df_filtered["options_list"] = df_filtered["options"]

        df_filtered["gt_option_str"] = df_filtered.apply(
            lambda row: row["options_list"][row["gt_idx"]], axis=1
        )
        # Assume sequences are always length 4 based on user confirmation
        for i in range(4):
            col_name = f"gt_obj_{i+1}"
            df_filtered[col_name] = df_filtered["gt_option_str"].apply(
                lambda x: x.split(", ")[i].strip() if len(x.split(", ")) > i else None
            )
            # Ensure None for missing objects if sequence is shorter than 4
            mask = df_filtered[col_name].isna()
            df_filtered.loc[mask, col_name] = None


    except Exception as e:
        print(f"Error parsing options or ground truth: {e}")
        return []

    # Drop rows where any GT object couldn't be parsed (shouldn't happen if all len=4)
    gt_cols = [f"gt_obj_{i+1}" for i in range(4)]
    df_filtered.dropna(subset=gt_cols, inplace=True)
    if df_filtered.empty:
        print("Warning: No valid GT sequences found after parsing.")
        return []


    # --- 2. Calculate Frequencies ---
    # 2a. Object-Position Frequencies
    position_freq = Counter()
    for i in range(1, 5): # Positions 1 to 4
        col_name = f"gt_obj_{i}"
        counts = df_filtered[col_name].value_counts()
        for obj, count in counts.items():
            position_freq[(obj, i)] += count

    # 2b. Adjacent Pair Frequencies
    pair_freq = Counter()
    # Pairs: (obj1, obj2), (obj2, obj3), (obj3, obj4)
    for i in range(1, 4):
        pair = tuple(zip(df_filtered[f"gt_obj_{i}"], df_filtered[f"gt_obj_{i+1}"]))
        pair_freq.update(pair)

    if not position_freq or not pair_freq:
        print("Warning: Could not calculate frequencies.")
        return []

    # --- 3. Normalize Frequencies ---
    pos_freq_values = np.array(list(position_freq.values())).reshape(-1, 1)
    pair_freq_values = np.array(list(pair_freq.values())).reshape(-1, 1)

    scaled_pos_freq_values = minmax_scale(pos_freq_values) if len(np.unique(pos_freq_values)) > 1 else np.ones_like(pos_freq_values)
    scaled_pair_freq_values = minmax_scale(pair_freq_values) if len(np.unique(pair_freq_values)) > 1 else np.ones_like(pair_freq_values)

    norm_pos_freq_map = {key: scaled_pos_freq_values[i][0] for i, key in enumerate(position_freq.keys())}
    norm_pair_freq_map = {key: scaled_pair_freq_values[i][0] for i, key in enumerate(pair_freq.keys())}

    # --- 4. Calculate Combined Bias Score ---
    bias_scores = []
    for _, row in df_filtered.iterrows():
        current_pos_score_sum = 0
        current_pair_score_sum = 0

        # Positional score component
        for i in range(1, 5):
            obj = row[f"gt_obj_{i}"]
            current_pos_score_sum += norm_pos_freq_map.get((obj, i), 0)

        # Pair score component
        for i in range(1, 4):
            pair = (row[f"gt_obj_{i}"], row[f"gt_obj_{i+1}"])
            current_pair_score_sum += norm_pair_freq_map.get(pair, 0)

        # Since all are length 4, no need to normalize by length here
        final_score = w_pos * current_pos_score_sum + w_pair * current_pair_score_sum
        bias_scores.append(final_score)

    df_filtered['bias_score'] = bias_scores

    # --- 5. Filter Based on Score ---
    df_sorted = df_filtered.sort_values('bias_score', ascending=False)
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids

import itertools # For generating pairs

def filter_app_order_scored_v3(
    df: pd.DataFrame,
    budget: int,
    w_pos: float = 1.0,          # Weight for positional frequency
    w_adj_pair: float = 1.0,     # Weight for adjacent pair frequency
    w_ordered_pair: float = 1.0  # Weight for general ordered pair frequency
    # min_samples: Optional constraint
) -> list:
    """
    Identify obj_appearance_order questions to eliminate based on a combined bias score.

    Scores questions based on:
    1. Frequency of ground truth objects in specific positions.
    2. Frequency of adjacent object pairs in the ground truth sequence.
    3. Frequency of all ordered object pairs (X before Y) in the ground truth sequence.

    Args:
        df: DataFrame with obj_appearance_order questions.
        budget: Total number of questions to eliminate.
        w_pos: Weight for positional frequency score component.
        w_adj_pair: Weight for adjacent pair frequency score component.
        w_ordered_pair: Weight for general ordered pair frequency component.

    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = df[df['question_type'] == 'obj_appearance_order'].copy()
    if df_filtered.empty:
        return []

    # --- 1. Extract Ground Truth Object Sequence ---
    try:
        df_filtered["gt_idx"] = df_filtered["ground_truth"].apply(lambda x: "ABCD".index(x))
        if isinstance(df_filtered["options"].iloc[0], str):
            df_filtered["options_list"] = df_filtered["options"].apply(
                lambda x: [opt.split('. ', 1)[1] for opt in x.split("'")[1::2]]
            )
        else:
            df_filtered["options_list"] = df_filtered["options"]

        df_filtered["gt_option_str"] = df_filtered.apply(
            lambda row: row["options_list"][row["gt_idx"]], axis=1
        )
        # Extract GT objects into a list column for easier processing
        df_filtered['gt_objects'] = df_filtered["gt_option_str"].apply(lambda x: [obj.strip() for obj in x.split(", ")[:4]]) # Assume len 4

        # Ensure all sequences have exactly 4 elements for consistency, padding if needed (though user said all are 4)
        # df_filtered['gt_objects'] = df_filtered['gt_objects'].apply(lambda x: x + [None]*(4-len(x)) if len(x)<4 else x[:4])

        # Drop rows if GT sequence extraction failed or resulted in < 4 objects
        df_filtered = df_filtered[df_filtered['gt_objects'].apply(lambda x: isinstance(x, list) and len(x) == 4 and all(i is not None for i in x))]


    except Exception as e:
        print(f"Error parsing options or ground truth: {e}")
        return []

    if df_filtered.empty:
        print("Warning: No valid GT sequences of length 4 found after parsing.")
        return []

    # --- 2. Calculate Frequencies ---
    position_freq = Counter()
    adj_pair_freq = Counter()
    ordered_pair_freq = Counter()

    sequences = df_filtered['gt_objects'].tolist()

    for seq in sequences:
        # Positional Frequencies
        for i, obj in enumerate(seq):
            position_freq[(obj, i + 1)] += 1 # Positions 1-4

        # Adjacent Pair Frequencies (Pairs: (o1,o2), (o2,o3), (o3,o4))
        for i in range(len(seq) - 1):
            adj_pair_freq[(seq[i], seq[i+1])] += 1

        # General Ordered Pair Frequencies (Pairs: (o1,o2), (o1,o3), (o1,o4), (o2,o3), (o2,o4), (o3,o4))
        for i, j in itertools.combinations(range(len(seq)), 2):
            ordered_pair_freq[(seq[i], seq[j])] += 1


    if not position_freq or not adj_pair_freq or not ordered_pair_freq:
        print("Warning: Could not calculate all necessary frequencies.")
        return []

    # --- 3. Normalize Frequencies ---
    pos_freq_values = np.array(list(position_freq.values())).reshape(-1, 1)
    adj_pair_freq_values = np.array(list(adj_pair_freq.values())).reshape(-1, 1)
    ordered_pair_freq_values = np.array(list(ordered_pair_freq.values())).reshape(-1, 1)

    scaled_pos_freq_values = minmax_scale(pos_freq_values) if len(np.unique(pos_freq_values)) > 1 else np.ones_like(pos_freq_values)
    scaled_adj_pair_freq_values = minmax_scale(adj_pair_freq_values) if len(np.unique(adj_pair_freq_values)) > 1 else np.ones_like(adj_pair_freq_values)
    scaled_ordered_pair_freq_values = minmax_scale(ordered_pair_freq_values) if len(np.unique(ordered_pair_freq_values)) > 1 else np.ones_like(ordered_pair_freq_values)

    norm_pos_freq_map = {key: scaled_pos_freq_values[i][0] for i, key in enumerate(position_freq.keys())}
    norm_adj_pair_freq_map = {key: scaled_adj_pair_freq_values[i][0] for i, key in enumerate(adj_pair_freq.keys())}
    norm_ordered_pair_freq_map = {key: scaled_ordered_pair_freq_values[i][0] for i, key in enumerate(ordered_pair_freq.keys())}

    # --- 4. Calculate Combined Bias Score ---
    bias_scores = []
    for seq in sequences: # Iterate through the sequences directly
        current_pos_score_sum = 0
        current_adj_pair_score_sum = 0
        current_ordered_pair_score_sum = 0

        # Positional score component
        for i, obj in enumerate(seq):
            current_pos_score_sum += norm_pos_freq_map.get((obj, i + 1), 0)

        # Adjacent pair score component
        for i in range(len(seq) - 1):
            current_adj_pair_score_sum += norm_adj_pair_freq_map.get((seq[i], seq[i+1]), 0)

        # Ordered pair score component
        for i, j in itertools.combinations(range(len(seq)), 2):
            current_ordered_pair_score_sum += norm_ordered_pair_freq_map.get((seq[i], seq[j]), 0)

        # Calculate final weighted score
        final_score = (
            w_pos * current_pos_score_sum +
            w_adj_pair * current_adj_pair_score_sum +
            w_ordered_pair * current_ordered_pair_score_sum
        )
        bias_scores.append(final_score)

    df_filtered['bias_score'] = bias_scores

    # --- 5. Filter Based on Score ---
    df_sorted = df_filtered.sort_values('bias_score', ascending=False)
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids

def get_app_order_scored_v4(
    df: pd.DataFrame,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_pair: float = 1.0,     # Weight for adjacent pair frequency component
    w_overall: float = 1.0   # Weight for overall object frequency component
) -> pd.DataFrame:
    qdf = df[df['question_type'] == 'obj_appearance_order'].copy()
    if qdf.empty:
        return pd.DataFrame(columns=df.columns)

    # --- 1. Extract Ground Truth Object Sequence ---
    qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
    qdf["gt_option"] = qdf.apply(lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1)

    gt_objects_list_flat = [] # For overall frequency
    # Assume sequences are always length 4
    for i in range(4):
        col_name = f"gt_obj_{i+1}"
        qdf[col_name] = qdf["gt_option"].apply(lambda x: x.split(", ")[i].strip() if len(x.split(", ")) > i else None)
        qdf.loc[qdf[col_name].isna(), col_name] = None
        gt_objects_list_flat.extend(qdf[col_name].dropna().tolist()) # Add to flat list

    gt_cols = [f"gt_obj_{i+1}" for i in range(4)]
    qdf.dropna(subset=gt_cols, inplace=True) # Ensure all 4 objects parsed
    if qdf.empty:
        print("Warning: No valid GT sequences found after parsing.")
        return []


    # --- 2. Calculate Frequencies ---
    # 2a. Object-Position Frequencies
    position_freq = Counter()
    for i in range(1, 5):
        counts = qdf[f"gt_obj_{i}"].value_counts()
        for obj, count in counts.items():
            position_freq[(obj, i)] += count

    # 2b. Adjacent Pair Frequencies
    pair_freq = Counter()
    for i in range(1, 4):
        pair = tuple(zip(qdf[f"gt_obj_{i}"], qdf[f"gt_obj_{i+1}"]))
        pair_freq.update(pair)

    # 2c. Overall Object Frequencies
    overall_obj_freq = Counter(gt_objects_list_flat)


    if not position_freq or not pair_freq or not overall_obj_freq:
        print("Warning: Could not calculate frequencies.")
        return []

    # --- 3. Normalize Frequencies ---
    pos_freq_values = np.array(list(position_freq.values())).reshape(-1, 1)
    pair_freq_values = np.array(list(pair_freq.values())).reshape(-1, 1)
    overall_freq_values = np.array(list(overall_obj_freq.values())).reshape(-1, 1)

    scaled_pos_freq_values = minmax_scale(pos_freq_values) if len(np.unique(pos_freq_values)) > 1 else np.ones_like(pos_freq_values)
    scaled_pair_freq_values = minmax_scale(pair_freq_values) if len(np.unique(pair_freq_values)) > 1 else np.ones_like(pair_freq_values)
    scaled_overall_freq_values = minmax_scale(overall_freq_values) if len(np.unique(overall_freq_values)) > 1 else np.ones_like(overall_freq_values)


    norm_pos_freq_map = {key: scaled_pos_freq_values[i][0] for i, key in enumerate(position_freq.keys())}
    norm_pair_freq_map = {key: scaled_pair_freq_values[i][0] for i, key in enumerate(pair_freq.keys())}
    norm_overall_freq_map = {key: scaled_overall_freq_values[i][0] for i, key in enumerate(overall_obj_freq.keys())}

    # --- 4. Calculate Combined Bias Score ---
    positional_scores = []
    pair_scores = []
    overall_scores = []

    for _, row in qdf.iterrows():
        current_pos_score_sum = 0
        current_pair_score_sum = 0
        current_overall_score_sum = 0

        gt_objs_in_row = [row[f"gt_obj_{i+1}"] for i in range(4)]

        # Positional score component
        for i, obj in enumerate(gt_objs_in_row):
            current_pos_score_sum += norm_pos_freq_map.get((obj, i + 1), 0)
            current_overall_score_sum += norm_overall_freq_map.get(obj, 0)  # Add overall score here

        # Pair score component
        for i in range(3):
            pair = (gt_objs_in_row[i], gt_objs_in_row[i + 1])
            current_pair_score_sum += norm_pair_freq_map.get(pair, 0)

        positional_scores.append(current_pos_score_sum)
        pair_scores.append(current_pair_score_sum)
        overall_scores.append(current_overall_score_sum)

    qdf['positional_score'] = minmax_scale(positional_scores)
    qdf['pair_score'] = minmax_scale(pair_scores)
    qdf['overall_score'] = minmax_scale(overall_scores)
    qdf['bias_score'] = (
        w_pos * qdf['positional_score'] +
        w_pair * qdf['pair_score'] +
        w_overall * qdf['overall_score']
    ) / 3 # Normalize by 3 components

    return qdf

def filter_app_order_scored_v4(
    df: pd.DataFrame,
    budget: int,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_pair: float = 1.0,     # Weight for adjacent pair frequency component
    w_overall: float = 1.0   # Weight for overall object frequency component
    # min_samples: Optional constraint can be added later if needed
) -> list:
    """
    Identify obj_appearance_order questions to eliminate based on a combined bias score.

    Scores questions based on:
    1. Frequency of their ground truth objects appearing in specific positions.
    2. Frequency of adjacent object pairs within the ground truth sequence.
    3. Overall frequency of their ground truth objects within this question type.

    Args:
        df: DataFrame with obj_appearance_order questions. Must contain
            'id', 'question_type', 'ground_truth', 'options' columns.
        budget: Total number of questions to eliminate.
        w_pos: Weight for positional frequency score component.
        w_pair: Weight for adjacent pair frequency score component.
        w_overall: Weight for overall object frequency score component.


    Returns:
        List of question IDs to eliminate.
    """
    if budget <= 0:
        return []

    df_filtered = get_app_order_scored_v4(df, w_pos=w_pos, w_pair=w_pair, w_overall=w_overall)
    if df_filtered.empty:
        return []

    # --- 5. Filter Based on Score ---
    df_sorted = df_filtered.sort_values('bias_score', ascending=False)
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids


def get_app_order_relative_score(
    df: pd.DataFrame,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_pair: float = 1.0,     # Weight for adjacent pair frequency component
    w_overall: float = 1.0   # Weight for overall object frequency component
) -> pd.DataFrame:
    """
    Calculates a relative bias score for obj_appearance_order questions.
    Score = score(GT_sequence) - max(score(Distractor_sequences)), where
    sequence scores are based on positional and adjacent pair frequencies.

    Args:
        df: DataFrame containing obj_appearance_order questions.
        w_pos: Weight for positional frequency component.
        w_pair: Weight for adjacent pair frequency component.
        w_overall: Weight for overall object frequency component.

    Returns:
        DataFrame with 'id' and 'relative_bias_score'.
    """
    qdf = df[df['question_type'] == 'obj_appearance_order'].copy()

    # --- 1. Extract All Object Sequences ---
    qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
    qdf["gt_option"] = qdf.apply(lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1)

    gt_objects_list_flat = [] # For overall frequency
    # Assume sequences are always length 4
    for i in range(4):
        gt_obj_i = qdf["gt_option"].apply(lambda x: x.split(", ")[i].strip())
        qdf[f"gt_obj_{i+1}"] = gt_obj_i # Add to dataframe
        gt_objects_list_flat.extend(gt_obj_i.dropna().tolist()) # Add to flat list

        qdf[f"opt_seq_{i+1}"] = qdf["options"].apply(lambda x: x[i].split(". ", 1)[1].split(", "))

    # --- 2. Calculate Frequencies (using only GT sequences for stats) ---
    position_freq = Counter()
    for i in range(1, 5):
        counts = qdf[f"gt_obj_{i}"].value_counts()
        for obj, count in counts.items(): position_freq[(obj, i)] += count
    pair_freq = Counter()
    for i in range(1, 4):
        pair = tuple(zip(qdf[f"gt_obj_{i}"], qdf[f"gt_obj_{i+1}"]))
        pair_freq.update(pair)
    comb_pair_freq = Counter()
    for i in range(1, 5):
        for j in range(i + 1, 5):
            pair = tuple(zip(qdf[f"gt_obj_{i}"], qdf[f"gt_obj_{j}"]))
            comb_pair_freq.update(pair)
    overall_obj_freq = Counter(gt_objects_list_flat)

    if not position_freq or not pair_freq:
        raise ValueError("No valid frequencies found.")

    # --- 3. Normalize Frequencies ---
    pos_freq_values = np.array(list(position_freq.values())).reshape(-1, 1)
    pair_freq_values = np.array(list(pair_freq.values())).reshape(-1, 1)
    comb_pair_freq_values = np.array(list(comb_pair_freq.values())).reshape(-1, 1)
    overall_freq_values = np.array(list(overall_obj_freq.values())).reshape(-1, 1)

    scaled_pos_freq_values = minmax_scale(pos_freq_values) if len(np.unique(pos_freq_values)) > 1 else np.ones_like(pos_freq_values)
    scaled_pair_freq_values = minmax_scale(pair_freq_values) if len(np.unique(pair_freq_values)) > 1 else np.ones_like(pair_freq_values)
    scaled_comb_pair_freq_values = minmax_scale(comb_pair_freq_values) if len(np.unique(comb_pair_freq_values)) > 1 else np.ones_like(comb_pair_freq_values)
    scaled_overall_freq_values = minmax_scale(overall_freq_values) if len(np.unique(overall_freq_values)) > 1 else np.ones_like(overall_freq_values)

    norm_pos_freq_map = {key: scaled_pos_freq_values[i][0] for i, key in enumerate(position_freq.keys())}
    norm_pair_freq_map = {key: scaled_pair_freq_values[i][0] for i, key in enumerate(pair_freq.keys())}
    norm_comb_pair_freq_map = {key: scaled_comb_pair_freq_values[i][0] for i, key in enumerate(comb_pair_freq.keys())}
    norm_overall_freq_map = {key: scaled_overall_freq_values[i][0] for i, key in enumerate(overall_obj_freq.keys())}

    # --- 4. Calculate Relative Bias Score ---
    bias_infos = []
    for index, row in qdf.iterrows():
        gt_idx = row['gt_idx']

        # gt_score = all_scores[gt_idx]
        # distractor_scores = [all_scores[i] for i in range(4) if i != gt_idx]
        # max_distractor_score = max(distractor_scores) if distractor_scores else 0

        # relative_score = gt_score - max_distractor_score
        bias_info = {'id': row['id']}
        max_distractor_pos_score = float('-inf')
        max_distractor_pair_score = float('-inf')
        max_distractor_comb_pair_score = float('-inf')
        max_distractor_overall_score = float('-inf')
        max_distractor_score = float('-inf')
        # for i, seq in enumerate(options_sequences):
        for i in range(4):
            seq = row[f"opt_seq_{i+1}"]
            # pos_score = pair_score = overall_score = 0
            pos_score = pair_score = comb_pair_score = overall_score = 0
            for j, obj in enumerate(seq):
                pos_score += norm_pos_freq_map.get((obj, j + 1), 0)
                overall_score += norm_overall_freq_map.get(obj, 0)
                if j < len(seq) - 1:
                    pair = (seq[j], seq[j + 1])
                    pair_score += norm_pair_freq_map.get(pair, 0)
                for k in range(j + 1, len(seq)):
                    pair = (seq[j], seq[k])
                    comb_pair_score += norm_comb_pair_freq_map.get(pair, 0)

            score = w_pos * pos_score + w_pair * pair_score + w_overall * overall_score / 3 # Normalize by 3 components
            bias_info[f'seq_{i}_pos_score'] = pos_score
            bias_info[f'seq_{i}_pair_score'] = pair_score
            bias_info[f'seq_{i}_comb_pair_score'] = comb_pair_score
            bias_info[f'seq_{i}_score'] = score
            if i == gt_idx:
                bias_info['gt_pos_score'] = pos_score
                bias_info['gt_pair_score'] = pair_score
                bias_info['gt_comb_pair_score'] = comb_pair_score
                bias_info['gt_obj_score'] = score
            else:
                max_distractor_pos_score = max(max_distractor_pos_score, pos_score)
                max_distractor_pair_score = max(max_distractor_pair_score, pair_score)
                max_distractor_comb_pair_score = max(max_distractor_comb_pair_score, comb_pair_score)
                max_distractor_score = max(max_distractor_score, score)

        # bias_info[f'seq_{i}_overall_score'] = overall_score
        # bias_info['gt_overall_score'] = overall_score
        # max_distractor_overall_score = max(max_distractor_overall_score, overall_score)
        # bias_info['max_distractor_overall_score'] = max_distractor_overall_score
        # bias_info['relative_bias_overall_score'] = bias_info['gt_overall_score'] - max_distractor_overall_score
        # overall score is same across all choices
        bias_info["overall_obj_score"] = overall_score

        bias_info['max_distractor_pos_score'] = max_distractor_pos_score
        bias_info['max_distractor_pair_score'] = max_distractor_pair_score
        bias_info['max_distractor_comb_pair_score'] = max_distractor_comb_pair_score
        bias_info['max_distractor_score'] = max_distractor_score

        bias_info['relative_bias_pos_score'] = bias_info['gt_pos_score'] - max_distractor_pos_score
        bias_info['relative_bias_pair_score'] = bias_info['gt_pair_score'] - max_distractor_pair_score
        bias_info['relative_bias_comb_pair_score'] = bias_info['gt_comb_pair_score'] - max_distractor_comb_pair_score
        # bias_info['relative_bias_score'] = bias_info['gt_obj_score'] - max_distractor_score
        # recalculate relative bias score using the differences in scores and then multiply by weights
        bias_info['relative_bias_score'] = (
            w_pos * bias_info['relative_bias_pos_score'] +
            w_pair * bias_info['relative_bias_pair_score'] +
            w_overall * overall_score
        ) / 3

        bias_infos.append(bias_info)

    # Create DataFrame from bias_infos
    bias_df = pd.DataFrame(bias_infos)

    # Merge with original DataFrame to keep all columns
    merged_df = pd.merge(qdf, bias_df, on='id', how='left')

    # Select and return relevant columns
    return merged_df


def filter_app_order_relative(
    df: pd.DataFrame,
    budget: int,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_pair: float = 1.0,      # Weight for adjacent pair frequency component
    w_overall: float = 1.0   # Weight for overall object frequency component
) -> list:
    """
    Filters obj_appearance_order questions based on the relative bias score.
    Removes questions where the GT sequence score is highest relative to distractors.

    Args:
        df_with_scores: DataFrame with 'id' and 'relative_bias_score'.
        budget: Max number of questions to remove.

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_app_order_relative_score(df, w_pos=w_pos, w_pair=w_pair, w_overall=w_overall)

    budget = min(budget, len(df_with_scores))
    if budget <= 0: return []

    # Sort by relative bias score (GT score much higher than distractors = higher score)
    df_sorted = df_with_scores.sort_values('relative_bias_score', ascending=False)

    remove_ids = df_sorted['id'].head(budget).tolist()
    return remove_ids

def get_app_order_relative_score_v2(
    df: pd.DataFrame,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_pair: float = 1.0,     # Weight for adjacent pair frequency component
    w_comb_pair: float = 1.0, # Weight for combined pair frequency component
) -> pd.DataFrame:
    """
    Calculates a relative bias score for obj_appearance_order questions.
    Score = score(GT_sequence) - max(score(Distractor_sequences)), where
    sequence scores are based on positional and adjacent pair frequencies.

    Args:
        df: DataFrame containing obj_appearance_order questions.
        w_pos: Weight for positional frequency component.
        w_pair: Weight for adjacent pair frequency component.
        w_comb_pair: Weight for combined pair frequency component.

    Returns:
        DataFrame with 'id' and 'relative_bias_score'.
    """
    qdf = df[df['question_type'] == 'obj_appearance_order'].copy()

    # --- 1. Extract All Object Sequences ---
    qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
    qdf["gt_option"] = qdf.apply(lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1)

    gt_objects_list_flat = [] # For overall frequency
    # Assume sequences are always length 4
    for i in range(4):
        gt_obj_i = qdf["gt_option"].apply(lambda x: x.split(", ")[i].strip())
        qdf[f"gt_obj_{i+1}"] = gt_obj_i # Add to dataframe
        gt_objects_list_flat.extend(gt_obj_i.dropna().tolist()) # Add to flat list

        qdf[f"opt_seq_{i+1}"] = qdf["options"].apply(lambda x: x[i].split(". ", 1)[1].split(", "))

    # --- 2. Calculate Frequencies (using only GT sequences for stats) ---
    position_freq = Counter()
    for i in range(1, 5):
        counts = qdf[f"gt_obj_{i}"].value_counts()
        for obj, count in counts.items(): position_freq[(obj, i)] += count
    pair_freq = Counter()
    for i in range(1, 4):
        pair = tuple(zip(qdf[f"gt_obj_{i}"], qdf[f"gt_obj_{i+1}"]))
        pair_freq.update(pair)
    comb_pair_freq = Counter()
    for i in range(1, 5):
        for j in range(i + 1, 5):
            pair = tuple(zip(qdf[f"gt_obj_{i}"], qdf[f"gt_obj_{j}"]))
            comb_pair_freq.update(pair)

    if not position_freq or not pair_freq:
        raise ValueError("No valid frequencies found.")

    # --- 3. Normalize Frequencies ---
    pos_freq_values = np.array(list(position_freq.values())).reshape(-1, 1)
    pair_freq_values = np.array(list(pair_freq.values())).reshape(-1, 1)
    comb_pair_freq_values = np.array(list(comb_pair_freq.values())).reshape(-1, 1)

    scaled_pos_freq_values = minmax_scale(pos_freq_values) if len(np.unique(pos_freq_values)) > 1 else np.ones_like(pos_freq_values)
    scaled_pair_freq_values = minmax_scale(pair_freq_values) if len(np.unique(pair_freq_values)) > 1 else np.ones_like(pair_freq_values)
    scaled_comb_pair_freq_values = minmax_scale(comb_pair_freq_values) if len(np.unique(comb_pair_freq_values)) > 1 else np.ones_like(comb_pair_freq_values)

    norm_pos_freq_map = {key: scaled_pos_freq_values[i][0] for i, key in enumerate(position_freq.keys())}
    norm_pair_freq_map = {key: scaled_pair_freq_values[i][0] for i, key in enumerate(pair_freq.keys())}
    norm_comb_pair_freq_map = {key: scaled_comb_pair_freq_values[i][0] for i, key in enumerate(comb_pair_freq.keys())}

    # --- 4. Calculate Relative Bias Score ---
    bias_infos = []
    for index, row in qdf.iterrows():
        gt_idx = row['gt_idx']

        # gt_score = all_scores[gt_idx]
        # distractor_scores = [all_scores[i] for i in range(4) if i != gt_idx]
        # max_distractor_score = max(distractor_scores) if distractor_scores else 0

        # relative_score = gt_score - max_distractor_score
        bias_info = {'id': row['id']}
        max_distractor_pos_score = float('-inf')
        max_distractor_pair_score = float('-inf')
        max_distractor_comb_pair_score = float('-inf')
        max_distractor_score = float('-inf')
        # for i, seq in enumerate(options_sequences):
        for i in range(4):
            seq = row[f"opt_seq_{i+1}"]
            pos_score = pair_score = comb_pair_score = 0
            for j, obj in enumerate(seq):
                pos_score += norm_pos_freq_map.get((obj, j + 1), 0)
                if j < len(seq) - 1:
                    pair = (seq[j], seq[j + 1])
                    pair_score += norm_pair_freq_map.get(pair, 0)
                for k in range(j + 1, len(seq)):
                    pair = (seq[j], seq[k])
                    comb_pair_score += norm_comb_pair_freq_map.get(pair, 0)

            score = w_pos * pos_score + w_pair * pair_score + w_comb_pair * comb_pair_score / 3 # Normalize by 3 components
            bias_info[f'seq_{i}_pos_score'] = pos_score
            bias_info[f'seq_{i}_pair_score'] = pair_score
            bias_info[f'seq_{i}_comb_pair_score'] = comb_pair_score
            bias_info[f'seq_{i}_score'] = score
            if i == gt_idx:
                bias_info['gt_pos_score'] = pos_score
                bias_info['gt_pair_score'] = pair_score
                bias_info['gt_comb_pair_score'] = comb_pair_score
                bias_info['gt_obj_score'] = score
            else:
                max_distractor_pos_score = max(max_distractor_pos_score, pos_score)
                max_distractor_pair_score = max(max_distractor_pair_score, pair_score)
                max_distractor_comb_pair_score = max(max_distractor_comb_pair_score, comb_pair_score)
                max_distractor_score = max(max_distractor_score, score)

        bias_info['max_distractor_pos_score'] = max_distractor_pos_score
        bias_info['max_distractor_pair_score'] = max_distractor_pair_score
        bias_info['max_distractor_comb_pair_score'] = max_distractor_comb_pair_score
        bias_info['max_distractor_score'] = max_distractor_score

        bias_info['relative_bias_pos_score'] = bias_info['gt_pos_score'] - max_distractor_pos_score
        bias_info['relative_bias_pair_score'] = bias_info['gt_pair_score'] - max_distractor_pair_score
        bias_info['relative_bias_comb_pair_score'] = bias_info['gt_comb_pair_score'] - max_distractor_comb_pair_score
        # bias_info['relative_bias_score'] = bias_info['gt_obj_score'] - max_distractor_score
        # recalculate relative bias score using the differences in scores and then multiply by weights
        bias_info['relative_bias_score'] = (
            w_pos * bias_info['relative_bias_pos_score'] +
            w_pair * bias_info['relative_bias_pair_score'] +
            w_comb_pair * bias_info['relative_bias_comb_pair_score']
        ) / 3

        bias_infos.append(bias_info)

    # Create DataFrame from bias_infos
    bias_df = pd.DataFrame(bias_infos)

    # Merge with original DataFrame to keep all columns
    merged_df = pd.merge(qdf, bias_df, on='id', how='left')

    # Select and return relevant columns
    return merged_df


def filter_app_order_relative_v2(
    df: pd.DataFrame,
    budget: int,
    w_pos: float = 1.0,      # Weight for positional frequency component
    w_pair: float = 1.0,      # Weight for adjacent pair frequency component
    w_comb_pair: float = 1.0   # Weight for combined pair frequency component
) -> list:
    """
    Filters obj_appearance_order questions based on the relative bias score.
    Removes questions where the GT sequence score is highest relative to distractors.

    Args:
        df: DataFrame containing obj_appearance_order questions.
        budget: Max number of questions to remove.
        w_pos: Weight for positional frequency component.
        w_pair: Weight for adjacent pair frequency component.
        w_comb_pair: Weight for combined pair frequency component.

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_app_order_relative_score_v2(df, w_pos=w_pos, w_pair=w_pair, w_comb_pair=w_comb_pair)

    budget = min(budget, len(df_with_scores))
    if budget <= 0: return []

    # Sort by relative bias score (GT score much higher than distractors = higher score)
    df_sorted = df_with_scores.sort_values('relative_bias_score', ascending=False)

    remove_ids = df_sorted['id'].head(budget).tolist()
    return remove_ids


from sklearn.preprocessing import minmax_scale, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError


# --- Helper to calculate score for a single sequence (from previous attempt) ---
def calculate_sequence_score_components(obj_list, norm_pos_freq_map, norm_pair_freq_map, norm_overall_freq_map):
    """Calculates individual score components for a sequence."""
    pos_score_sum = 0
    pair_score_sum = 0
    overall_score_sum = 0
    if len(obj_list) != 4 or any(pd.isna(o) for o in obj_list):
        return 0.0, 0.0, 0.0 # Handle incomplete/invalid sequences

    # Positional score component
    for i, obj in enumerate(obj_list):
        pos_score_sum += norm_pos_freq_map.get((obj, i + 1), 0)
        overall_score_sum += norm_overall_freq_map.get(obj, 0)
    # Pair score component
    for i in range(3):
        pair = (obj_list[i], obj_list[i+1])
        pair_score_sum += norm_pair_freq_map.get(pair, 0)
    return pos_score_sum, pair_score_sum, overall_score_sum

def get_app_order_rf_score(df: pd.DataFrame, n_splits=5, seed=42) -> pd.DataFrame:
    """
    Calculates a bias score for obj_appearance_order questions using the
    prediction probability from a Random Forest trained on statistical features.

    Args:
        df: DataFrame containing obj_appearance_order questions.
        n_splits: Number of folds for cross-validation prediction.
        seed: Random seed.

    Returns:
        DataFrame with 'id' and 'bias_score' (RF probability for the true class).
    """
    w_pos = 1.0
    w_pair = 1.0
    w_overall = 1.0

    qdf = df[df['question_type'] == 'obj_appearance_order'].copy()

    # --- 1. Extract All Object Sequences ---
    qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
    qdf["gt_option"] = qdf.apply(lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1)

    gt_objects_list_flat = [] # For overall frequency
    # Assume sequences are always length 4
    for i in range(4):
        gt_obj_i = qdf["gt_option"].apply(lambda x: x.split(", ")[i].strip())
        qdf[f"gt_obj_{i+1}"] = gt_obj_i # Add to dataframe
        gt_objects_list_flat.extend(gt_obj_i.dropna().tolist()) # Add to flat list

        qdf[f"opt_seq_{i+1}"] = qdf["options"].apply(lambda x: x[i].split(". ", 1)[1].split(", "))

    # --- 2. Calculate Frequencies (using only GT sequences for stats) ---
    position_freq = Counter()
    for i in range(1, 5):
        counts = qdf[f"gt_obj_{i}"].value_counts()
        for obj, count in counts.items(): position_freq[(obj, i)] += count
    pair_freq = Counter()
    for i in range(1, 4):
        pair = tuple(zip(qdf[f"gt_obj_{i}"], qdf[f"gt_obj_{i+1}"]))
        pair_freq.update(pair)
    comb_pair_freq = Counter()
    for i in range(1, 5):
        for j in range(i + 1, 5):
            pair = tuple(zip(qdf[f"gt_obj_{i}"], qdf[f"gt_obj_{j}"]))
            comb_pair_freq.update(pair)
    overall_obj_freq = Counter(gt_objects_list_flat)

    if not position_freq or not pair_freq:
        raise ValueError("No valid frequencies found.")

    # --- 3. Normalize Frequencies ---
    pos_freq_values = np.array(list(position_freq.values())).reshape(-1, 1)
    pair_freq_values = np.array(list(pair_freq.values())).reshape(-1, 1)
    comb_pair_freq_values = np.array(list(comb_pair_freq.values())).reshape(-1, 1)
    overall_freq_values = np.array(list(overall_obj_freq.values())).reshape(-1, 1)

    scaled_pos_freq_values = minmax_scale(pos_freq_values) if len(np.unique(pos_freq_values)) > 1 else np.ones_like(pos_freq_values)
    scaled_pair_freq_values = minmax_scale(pair_freq_values) if len(np.unique(pair_freq_values)) > 1 else np.ones_like(pair_freq_values)
    scaled_comb_pair_freq_values = minmax_scale(comb_pair_freq_values) if len(np.unique(comb_pair_freq_values)) > 1 else np.ones_like(comb_pair_freq_values)
    scaled_overall_freq_values = minmax_scale(overall_freq_values) if len(np.unique(overall_freq_values)) > 1 else np.ones_like(overall_freq_values)

    norm_pos_freq_map = {key: scaled_pos_freq_values[i][0] for i, key in enumerate(position_freq.keys())}
    norm_pair_freq_map = {key: scaled_pair_freq_values[i][0] for i, key in enumerate(pair_freq.keys())}
    norm_comb_pair_freq_map = {key: scaled_comb_pair_freq_values[i][0] for i, key in enumerate(comb_pair_freq.keys())}
    norm_overall_freq_map = {key: scaled_overall_freq_values[i][0] for i, key in enumerate(overall_obj_freq.keys())}

    # --- 4. Calculate Relative Bias Score ---
    bias_infos = []
    for index, row in qdf.iterrows():
        gt_idx = row['gt_idx']

        # gt_score = all_scores[gt_idx]
        # distractor_scores = [all_scores[i] for i in range(4) if i != gt_idx]
        # max_distractor_score = max(distractor_scores) if distractor_scores else 0

        # relative_score = gt_score - max_distractor_score
        bias_info = {'id': row['id']}
        # max_distractor_pos_score = float('-inf')
        # max_distractor_pair_score = float('-inf')
        # max_distractor_comb_pair_score = float('-inf')
        # max_distractor_overall_score = float('-inf')
        # max_distractor_score = float('-inf')
        distractor_pos_scores = []
        distractor_pair_scores = []
        distractor_comb_pair_scores = []
        distractor_overall_scores = []
        distractor_scores = []
        # for i, seq in enumerate(options_sequences):
        for i in range(4):
            seq = row[f"opt_seq_{i+1}"]
            # pos_score = pair_score = overall_score = 0
            pos_score = pair_score = comb_pair_score = overall_score = 0
            for j, obj in enumerate(seq):
                pos_score += norm_pos_freq_map.get((obj, j + 1), 0)
                overall_score += norm_overall_freq_map.get(obj, 0)
                if j < len(seq) - 1:
                    pair = (seq[j], seq[j + 1])
                    pair_score += norm_pair_freq_map.get(pair, 0)
                for k in range(j + 1, len(seq)):
                    pair = (seq[j], seq[k])
                    comb_pair_score += norm_comb_pair_freq_map.get(pair, 0)

            score = w_pos * pos_score + w_pair * pair_score + w_overall * overall_score / 3 # Normalize by 3 components
            bias_info[f'seq_{i}_pos_score'] = pos_score
            bias_info[f'seq_{i}_pair_score'] = pair_score
            bias_info[f'seq_{i}_overall_score'] = overall_score
            bias_info[f'seq_{i}_comb_pair_score'] = comb_pair_score
            bias_info[f'seq_{i}_score'] = score
            if i == gt_idx:
                bias_info['gt_pos_score'] = pos_score
                bias_info['gt_pair_score'] = pair_score
                bias_info['gt_overall_score'] = overall_score
                bias_info['gt_comb_pair_score'] = comb_pair_score
                bias_info['gt_obj_score'] = score
            else:
                # max_distractor_pos_score = max(max_distractor_pos_score, pos_score)
                # max_distractor_pair_score = max(max_distractor_pair_score, pair_score)
                # max_distractor_overall_score = max(max_distractor_overall_score, overall_score)
                # max_distractor_comb_pair_score = max(max_distractor_comb_pair_score, comb_pair_score)
                # max_distractor_score = max(max_distractor_score, score)
                distractor_pos_scores.append(pos_score)
                distractor_pair_scores.append(pair_score)
                distractor_overall_scores.append(overall_score)
                distractor_comb_pair_scores.append(comb_pair_score)
                distractor_scores.append(score)
        max_distractor_pos_score = max(distractor_pos_scores)
        max_distractor_pair_score = max(distractor_pair_scores)
        max_distractor_overall_score = max(distractor_overall_scores)
        max_distractor_comb_pair_score = max(distractor_comb_pair_scores)
        max_distractor_score = max(distractor_scores)

        mean_distractor_pos_score = np.mean(distractor_pos_scores)
        mean_distractor_pair_score = np.mean(distractor_pair_scores)
        mean_distractor_overall_score = np.mean(distractor_overall_scores)
        mean_distractor_comb_pair_score = np.mean(distractor_comb_pair_scores)
        mean_distractor_score = np.mean(distractor_scores)

        bias_info['max_distractor_pos_score'] = max_distractor_pos_score
        bias_info['max_distractor_pair_score'] = max_distractor_pair_score
        bias_info['max_distractor_overall_score'] = max_distractor_overall_score
        bias_info['max_distractor_comb_pair_score'] = max_distractor_comb_pair_score
        bias_info['max_distractor_score'] = max_distractor_score

        bias_info['mean_distractor_pos_score'] = mean_distractor_pos_score
        bias_info['mean_distractor_pair_score'] = mean_distractor_pair_score
        bias_info['mean_distractor_overall_score'] = mean_distractor_overall_score
        bias_info['mean_distractor_comb_pair_score'] = mean_distractor_comb_pair_score
        bias_info['mean_distractor_score'] = mean_distractor_score

        bias_info['relative_bias_pos_score'] = bias_info['gt_pos_score'] - max_distractor_pos_score
        bias_info['relative_bias_pair_score'] = bias_info['gt_pair_score'] - max_distractor_pair_score
        bias_info['relative_bias_overall_score'] = bias_info['gt_overall_score'] - max_distractor_overall_score
        bias_info['relative_bias_comb_pair_score'] = bias_info['gt_comb_pair_score'] - max_distractor_comb_pair_score
        bias_info['relative_bias_score'] = bias_info['gt_obj_score'] - max_distractor_score

        bias_info['relative_bias_pos_score_mean'] = bias_info['gt_pos_score'] - mean_distractor_pos_score
        bias_info['relative_bias_pair_score_mean'] = bias_info['gt_pair_score'] - mean_distractor_pair_score
        bias_info['relative_bias_overall_score_mean'] = bias_info['gt_overall_score'] - mean_distractor_overall_score
        bias_info['relative_bias_comb_pair_score_mean'] = bias_info['gt_comb_pair_score'] - mean_distractor_comb_pair_score
        bias_info['relative_bias_score_mean'] = bias_info['gt_obj_score'] - mean_distractor_score

        bias_infos.append(bias_info)

    # Create DataFrame from bias_infos
    bias_df = pd.DataFrame(bias_infos).set_index('id')

    # Define feature columns (all calculated numeric scores)
    feature_cols = [col for col in bias_df.columns if col != 'id']


    # --- 5. Train RF and Get Probabilities ---
    X = bias_df[feature_cols]
    # Encode target labels ('A', 'B', 'C', 'D') numerically
    le = LabelEncoder()
    y = le.fit_transform(qdf['ground_truth'].values)
    label_map = dict(zip(le.classes_, le.transform(le.classes_))) # Map 'A'->0 etc.

    # Define CV strategy - ensure n_splits is not > smallest class count
    min_class_count = np.min(np.bincount(y))
    actual_n_splits = min(n_splits, min_class_count)
    if actual_n_splits < n_splits:
        print(f"Warning: Reduced n_splits from {n_splits} to {actual_n_splits} due to small class size ({min_class_count}).")
    if actual_n_splits < 2:
        print("Error: Cannot perform cross-validation with less than 2 splits.")
        # Fallback: Train on all data and predict (less ideal)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf.fit(X, y)
        all_probas = rf.predict_proba(X)
        print("Warning: Using probabilities from model trained on full data due to CV limitations.")
    else:
        # Use StratifiedKFold to ensure class distribution is maintained across folds
        cv = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=seed)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        print(f"Performing {actual_n_splits}-Fold CV to get probabilities...")
        try:
            all_probas = cross_val_predict(rf, X, y, cv=cv, method='predict_proba', n_jobs=-1)
            print("Cross-validation prediction complete.")
        except Exception as e:
            print(f"Error during cross_val_predict: {e}. Trying non-parallel.")
            # Try without n_jobs=-1 if the backend causes issues
            all_probas = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')
            print("Cross-validation prediction complete (non-parallel).")

    # Extract probability of the TRUE class for each sample
    true_class_indices = [label_map[gt] for gt in qdf['ground_truth'].values]
    bias_scores = [all_probas[i, true_class_indices[i]] for i in range(len(all_probas))]

    merged_df = pd.merge(qdf, bias_df, on='id', how='left')

    merged_df['bias_score'] = bias_scores
    return merged_df

def filter_app_order_rf(
    df: pd.DataFrame,
    budget: int,
    seed: int = 42
) -> list:
    """
    Filters obj_appearance_order questions based on the RF prediction probability
    score. Removes questions where the model was most confident in predicting
    the correct answer from statistical features alone.

    Args:
        df_with_scores: DataFrame with 'id' and 'bias_score' (RF probability).
        budget: Max number of questions to remove.

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_app_order_rf_score(df, seed=seed)

    # Sort by bias score (highest probability = most biased = remove first)
    df_sorted = df_with_scores.sort_values('bias_score', ascending=False)

    remove_ids = df_sorted['id'].head(budget).tolist()
    return remove_ids

def filter_route_planning_v1(df, budget):
    """
    Identify route_planning questions to eliminate.

    Args:
        df: DataFrame with route_planning questions
        budget: Number of questions to eliminate

    Returns:
        List of question IDs to eliminate
    """
    # Count frequency of each answer
    answer_counts = df['ground_truth'].value_counts()

    # Calculate target distribution (ideally uniform)
    total_questions = len(df)
    num_answers = len(answer_counts)
    target_per_answer = total_questions / num_answers

    # Initialize list of IDs to remove
    remove_ids = []

    # Start with most frequent answers
    for answer in answer_counts.index:
        if len(remove_ids) >= budget:
            break

        answer_questions = df[
            (df['ground_truth'] == answer) &
            (~df['id'].isin(remove_ids))
        ]

        # Calculate excess above target
        excess = max(0, len(answer_questions) - target_per_answer)
        to_remove = min(int(excess), budget - len(remove_ids))

        if to_remove > 0:
            questions_to_remove = answer_questions.sample(to_remove, random_state=42)
            remove_ids.extend(questions_to_remove['id'].tolist())

    return remove_ids[:budget]  # Ensure we don't exceed budget

def get_route_planning_score(
        df: pd.DataFrame,
        w_val_freq: float = 1.0,  # Weight for value frequency component
        w_opt_freq: float = 1.0    # Weight for option frequency component
    ) -> pd.DataFrame:
    """
    Calculates bias scores for route_planning questions based on the
    frequency of the ground truth answer string.

    Args:
        df: DataFrame containing route_planning questions.
            Requires 'id', 'question_type', 'ground_truth'.

    Returns:
        DataFrame with 'id' and 'bias_score' columns.
    """
    qdf = df[df['question_type'] == 'route_planning'].copy()

    # --- 1. Calculate Frequency of GT Answer Strings ---
    # Ensure GT is string, handle potential non-string types if necessary

    qdf["gt_idx"] = qdf["ground_truth"].apply(lambda x: "ABCD".index(x))
    qdf["gt_option"] = qdf.apply(lambda row: row["options"][row["gt_idx"]].split(". ")[-1], axis=1)

    gt_counts = qdf['gt_option'].value_counts()
    gt_letter_counts = qdf['ground_truth'].value_counts()

    # --- 2. Calculate Bias Score (Normalized Frequency) ---
    # Map counts to each row
    qdf['gt_val_freq'] = qdf['gt_option'].map(gt_counts)
    qdf['gt_val_freq'].fillna(0, inplace=True) # Handle cases if somehow GT isn't in counts

    qdf["gt_opt_freq"] = qdf["ground_truth"].map(gt_letter_counts)
    qdf["gt_opt_freq"].fillna(0, inplace=True) # Handle cases if somehow GT isn't in counts

    # Normalize frequency to get bias score
    qdf['val_freq_score'] = minmax_scale(qdf['gt_val_freq'])
    qdf['opt_freq_score'] = minmax_scale(qdf['gt_opt_freq'])
    qdf['bias_score'] = (
        w_val_freq * qdf['val_freq_score'] +
        w_opt_freq * qdf['opt_freq_score']
    ) / 2  # Normalize by number of components

    return qdf

def filter_route_planning_gt_freq(
    df: pd.DataFrame,
    budget: int,
    w_val_freq: float = 1.0,  # Weight for value frequency component
    w_opt_freq: float = 1.0    # Weight for option frequency component
) -> list:
    """
    Filters route_planning questions by removing those with the highest
    bias scores (most frequent ground truth answers).

    Args:
        df: DataFrame with route_planning questions
        budget: The maximum number of questions to remove.

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_route_planning_score(df, w_val_freq=w_val_freq, w_opt_freq=w_opt_freq)
    if df_with_scores.empty:
        return []

    # Ensure budget is valid
    budget = min(budget, len(df_with_scores))
    if budget <= 0:
        return []

    # Sort by bias score (highest frequency first)
    df_sorted = df_with_scores.sort_values('bias_score', ascending=False)

    # Select the top 'budget' IDs for removal (hard truncation)
    remove_ids = df_sorted['id'].head(budget).tolist()

    return remove_ids

def get_route_planning_score_v2(
    df: pd.DataFrame,
    w_obj_freq: float = 1.0,    # Weight for object frequency
    w_route_freq: float = 1.0,  # Weight for route string frequency
    w_steps_dist: float = 1.0   # Weight for typicality of step count
    ) -> pd.DataFrame:
    """
    Calculates primary grouping key (gt_answer_letter) and a secondary bias score
    for route_planning questions. Secondary score based on object frequencies,
    route string frequency, and typicality of step count.

    Args:
        df: DataFrame containing route_planning questions.
        w_obj_freq: Weight for object frequency component of secondary score.
        w_route_freq: Weight for route string frequency component.
        w_steps_dist: Weight for step count typicality component.

    Returns:
        DataFrame with 'id', 'gt_answer_letter', and 'secondary_bias_score'.
    """
    qdf = df[df['question_type'] == 'route_planning'].copy()

    # --- 1. Extract Features ---
    # Extract objects
    qdf[["beginning_object", "facing_object", "target_object"]] = \
        qdf["question"].str.extract(r'You are a robot beginning (?:at|by) the (.*?) (?:facing the|facing to|facing towards the|facing|with your back to the) (.*?)\. You want to navigate to the (.*?)\.')
    for col in ["beginning_object", "facing_object", "target_object"]:
        qdf[col] = qdf[col].str.replace(r" and$", "", regex=True).str.replace(r"^the ", "", regex=True).str.strip()

    # Extract GT Letter and Route String
    qdf["gt_answer_letter"] = qdf["ground_truth"] # Already A/B/C/D
    qdf["gt_idx"] = qdf["gt_answer_letter"].apply(lambda x: "ABCD".index(x))
    qdf["gt_route_str"] = qdf.apply(lambda row: row["options"][row["gt_idx"]].strip(), axis=1)

    # Extract Num Steps
    qdf["num_gt_steps"] = qdf["gt_route_str"].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)

    # Drop rows where extraction failed
    qdf.dropna(subset=["beginning_object", "facing_object", "target_object", "gt_route_str", "gt_answer_letter"], inplace=True)

    # --- 2. Calculate Frequencies & Stats ---
    # Object Frequencies
    all_objects = pd.concat([qdf["beginning_object"], qdf["facing_object"], qdf["target_object"]]).dropna().tolist()
    obj_freq = Counter(all_objects)
    # Route String Frequencies
    route_freq = qdf['gt_route_str'].value_counts()
    # Step Count Stats
    mean_steps = qdf['num_gt_steps'].mean()
    std_steps = qdf['num_gt_steps'].std()
    epsilon = 1e-6

    # --- 3. Normalize Frequencies / Calculate Score Components ---
    # Object Freq Score (sum of normalized freq for the 3 objects)
    norm_obj_freq_map = {k: v / len(all_objects) for k, v in obj_freq.items()} # Simple normalization
    qdf['obj_freq_score_comp'] = qdf.apply(
        lambda row: norm_obj_freq_map.get(row["beginning_object"], 0) + \
                    norm_obj_freq_map.get(row["facing_object"], 0) + \
                    norm_obj_freq_map.get(row["target_object"], 0),
        axis=1)

    # Route Freq Score (normalized freq of the route string)
    qdf['route_freq_map'] = qdf['gt_route_str'].map(route_freq)
    qdf['route_freq_score_comp'] = minmax_scale(qdf['route_freq_map'].fillna(0))

    # Step Count Typicality Score (1 - normalized distance from mean)
    qdf['steps_dist_norm'] = abs(qdf['num_gt_steps'] - mean_steps) / (std_steps + epsilon)
    qdf['steps_dist_score_comp'] = 1.0 - minmax_scale(qdf['steps_dist_norm'] + epsilon)

    # --- 4. Calculate Final Secondary Bias Score ---
    qdf['bias_score'] = (
        w_obj_freq * qdf['obj_freq_score_comp'] +
        w_route_freq * qdf['route_freq_score_comp'] +
        w_steps_dist * qdf['steps_dist_score_comp']
    )

    return qdf

def filter_route_planning_v2(
    df: pd.DataFrame,
    budget: int, # Still potentially ignored if balancing requires more/less
    w_obj_freq: float = 1.0,    # Weight for object frequency
    w_route_freq: float = 1.0,  # Weight for route string frequency
    w_steps_dist: float = 1.0,   # Weight for typicality of step count
    verbose: bool = False
) -> list:
    """
    Filters route_planning questions by balancing the ground truth answer letter
    distribution ('A'/'B'/'C'/'D'), prioritizing removal of samples with a
    higher secondary bias score (based on object/route freq, step typicality).

    Args:
        df_with_scores: DataFrame containing 'id', 'gt_answer_letter',
                        and 'secondary_bias_score' for route_planning questions.
        budget: The maximum target number of questions to remove.
                (Current implementation prioritizes balancing).

    Returns:
        List of question IDs selected for elimination.
    """
    df_with_scores = get_route_planning_score_v2(df, w_obj_freq=w_obj_freq, w_route_freq=w_route_freq, w_steps_dist=w_steps_dist)

    final_remove_ids = set()
    total_candidates_for_balance = 0 # Just for reporting

    gt_counts = df_with_scores['gt_answer_letter'].value_counts()
    if len(gt_counts) <= 1:
        return [] # Cannot balance

    min_count = gt_counts.min()
    if verbose:
        print(f"Route Planning Min count: {min_count}, Counts: {gt_counts.to_dict()}")

    gt_to_ids_to_remove = {}

    for gt_answer_letter, count in gt_counts.items():
        num_to_remove = count - min_count
        if num_to_remove > 0:
            total_candidates_for_balance += num_to_remove
            # Get samples for this specific ground truth answer letter
            gt_samples = df_with_scores[df_with_scores['gt_answer_letter'] == gt_answer_letter]
            # Sort them by secondary_bias_score (highest first)
            gt_samples_sorted = gt_samples.sort_values('bias_score', ascending=False)
            # Select the top 'num_to_remove' IDs based on score
            ids_to_remove_for_gt = gt_samples_sorted['id'].head(num_to_remove).tolist()
            final_remove_ids.update(ids_to_remove_for_gt)
            gt_to_ids_to_remove[gt_answer_letter] = ids_to_remove_for_gt

    if verbose:
        print(f"Total route_planning samples identified for full balancing: {total_candidates_for_balance}")
        print(f"Actual number selected (prioritizing high secondary score): {len(final_remove_ids)}")

    # Optional: Enforce budget strictly
    if len(final_remove_ids) > budget:
        remove_ids = []
        while len(remove_ids) < budget:
            # pop the highest scoring id from the gt with the most ids left
            gt_answer_letter = max(gt_to_ids_to_remove, key=lambda k: len(gt_to_ids_to_remove[k]))
            id = gt_to_ids_to_remove[gt_answer_letter].pop(0)
            remove_ids.append(id)

        final_remove_ids = remove_ids

    return list(final_remove_ids)


### ---Overall Debiasing Functions --- ###


# aliases for various versions of the same function
filter_object_size_estimation = filter_object_size_estimation_scored_v2
filter_object_counting = filter_object_counting_dynamic_pct
filter_obj_appearance_order = filter_app_order_relative_v2
filter_object_abs_distance = filter_object_abs_distance_scored_log_global
filter_object_rel_distance = filter_object_rel_distance_score_v2
filter_object_rel_direction = filter_object_rel_direction_v2
filter_room_size_estimation = filter_room_size_estimation_sampled_pdf
filter_route_planning = filter_route_planning_v2

filter_functions = {
    'object_size_estimation': filter_object_size_estimation,
    'object_counting': filter_object_counting,
    'object_abs_distance': filter_object_abs_distance,
    'room_size_estimation': filter_room_size_estimation,
    'object_rel_distance': filter_object_rel_distance,
    'object_rel_direction_easy': filter_object_rel_direction,
    'object_rel_direction_medium': filter_object_rel_direction,
    'object_rel_direction_hard': filter_object_rel_direction,
    'obj_appearance_order': filter_obj_appearance_order,
    'route_planning': filter_route_planning
}

budgets = {
    "object_size_estimation": 600,
    "object_abs_distance": 400,
    "object_rel_distance": 400,
    "obj_appearance_order": 300,
    "object_counting": 314,
    # "object_rel_direction_medium": 324,
    # "object_rel_direction_hard": 257,
    # "object_rel_direction_easy": 5,
    "object_rel_direction": 324 + 257 + 5,
    "room_size_estimation": 88,
    "route_planning": 80,
}

def debias_vsibench(save_path: str = "data/removed_ids.txt"):
    print("Debiasing VSIBench...")

    sep = "=" * 100

    # Load the data with scores
    print("Loading data and evaluating reference models...")
    print(sep)

    df = get_vsi_with_scores()

    # Filter the data based on the budgets
    print(sep)
    print("Filtering data based on budgets...")
    print(f"Budgets: {json.dumps(budgets, indent=4)}")
    ids_obj_size = filter_object_size_estimation(df, budget=budgets["object_size_estimation"])
    ids_abs_distance = filter_object_abs_distance(df, budget=budgets["object_abs_distance"])
    ids_rel_distance = filter_object_rel_distance(df, budget=budgets["object_rel_distance"])
    ids_obj_app_order = filter_obj_appearance_order(df, budget=budgets["obj_appearance_order"])
    ids_obj_count = filter_object_counting(df, budget=budgets["object_counting"])
    ids_rel_dir = filter_object_rel_direction(df, budget=budgets["object_rel_direction"])
    ids_room_size = filter_room_size_estimation(df, budget=budgets["room_size_estimation"])
    ids_route_planning = filter_route_planning(df, budget=budgets["route_planning"])

    ids_map = {
        "object_size_estimation": ids_obj_size,
        "object_abs_distance": ids_abs_distance,
        "object_rel_distance": ids_rel_distance,
        "obj_appearance_order": ids_obj_app_order,
        "object_counting": ids_obj_count,
        "object_rel_direction": ids_rel_dir,
        "room_size_estimation": ids_room_size,
        "route_planning": ids_route_planning,
    }

    all_ids = set()
    print(sep)
    for k, v in ids_map.items():
        all_ids = all_ids.union(set(v))
        n_removed = len(v)
        n_orig = len(df[df["question_type"].str.startswith(k)])
        n_remaining = n_orig - len(v)
        print(f"{n_removed} / {n_orig} ({n_removed/n_orig * 100:.2f}%) \t of {k} removed{' ' * (25 - len(k))}\t => {n_remaining} / {n_orig} ({n_remaining/n_orig * 100:.2f}%) remaining")

    print(sep)
    print(f"Number of ids removed: {len(all_ids)} / {len(df)} ({len(all_ids)/len(df) * 100:.2f}%)")
    print(f"Number of ids remaining: {len(df) - len(all_ids)} / {len(df)} ({(len(df) - len(all_ids))/len(df) * 100:.2f}%)")
    print(sep)

    df["removed"] = df["id"].isin(all_ids)
    deb_df = df.loc[~df["removed"]]
    removed_df = df.loc[df["removed"]]

    # Calculate scores for each model
    rem_scores = evaluate_models(removed_df).T["overall"]
    deb_scores = evaluate_models(deb_df).T["overall"]
    orig_scores = evaluate_models(df).T["overall"]
    diff_scores = deb_scores - orig_scores

    # combine the series into a dataframe
    scores_df = pd.DataFrame({
        "Removed": rem_scores,
        "Original": orig_scores,
        "Debiased": deb_scores,
        "Difference": diff_scores
    })
    scores_df = scores_df.reset_index().sort_values(by="Difference", ascending=True)
    scores_df = scores_df.rename(columns={"index": "Model"})

    print("Overall scores for each model:")
    print(scores_df)
    print(sep)

    # save the removed ids to a txt file
    print(f"Saving removed ids to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for id in all_ids:
            f.write(f"{id}\n")
    print(f"Finished saving removed ids to {save_path}")
    return df, removed_df, deb_df, scores_df


if __name__ == "__main__":
    debias_vsibench()