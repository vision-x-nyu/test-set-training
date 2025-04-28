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
from scipy.stats import lognorm

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
    if len(final_remove_ids) > budget:
        if verbose:
            print(f"Warning: Balancing requires {len(final_remove_ids)} removals, exceeding budget {budget}. Selecting subset.")
        final_remove_ids_list = list(final_remove_ids)
        random.shuffle(final_remove_ids_list)
        final_remove_ids = set(final_remove_ids_list[:budget])

    return list(final_remove_ids)

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