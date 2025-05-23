import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, minmax_scale

# =============================================================================
# 0.  LOAD DATASET -------------------------------------------------------------
# =============================================================================

vsibench = load_dataset("nyu-visionx/VSI-Bench")
df_full = vsibench["test"].to_pandas()

# =============================================================================
# 1.  COMMON HELPERS -----------------------------------------------------------
# =============================================================================

def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Label‑encode object columns in *place* (train & test)."""
    cat_cols = X_train.select_dtypes(include="object").columns
    encoders = {}
    for col in cat_cols:
        enc = LabelEncoder().fit(pd.concat([X_train[col], X_test[col]], axis=0).astype(str))
        X_train[col] = enc.transform(X_train[col].astype(str))
        X_test[col]  = enc.transform(X_test[col].astype(str))
        encoders[col] = enc
    return encoders

# =============================================================================
# 2.  OBJECT‑REL‑DISTANCE BIAS MODEL (unchanged) ------------------------------
# =============================================================================

# Pre‑extract once (cheap) -----------------------------------------------------
qdf_rel = df_full[df_full["question_type"] == "object_rel_distance"].copy()
rel_regex = r"which of these objects \((.*?), (.*?), (.*?), (.*?)\) is the closest to the (.*?)\?$"
qdf_rel[["object_1", "object_2", "object_3", "object_4", "target_object"]] = qdf_rel["question"].str.extract(rel_regex)
qdf_rel["gt_idx"] = qdf_rel["ground_truth"].apply(lambda x: "ABCD".index(x))
qdf_rel["gt_option"] = qdf_rel.apply(lambda r: r["options"][r["gt_idx"]], axis=1)
qdf_rel["gt_object"] = qdf_rel["gt_option"].apply(lambda s: s.split(". ")[-1].strip())
qdf_rel["tgt_gt_pair"] = qdf_rel.apply(lambda r: "-".join(sorted([r["target_object"], r["gt_object"]])), axis=1)
qdf_rel["tgt_gt_ord_pair"] = qdf_rel.apply(lambda r: f"{r['target_object']}-{r['gt_object']}", axis=1)

rel_feature_cols = [
    'object_1', 'object_2', 'object_3', 'object_4', 'target_object',
    'max_option_freq',
    'max_tgt_option_pair_freq', 'max_tgt_option_ord_pair_freq',
    'opt_0_option_freq', 'opt_0_tgt_option_pair_freq', 'opt_0_tgt_option_ord_pair_freq',
    'opt_1_option_freq', 'opt_1_tgt_option_pair_freq', 'opt_1_tgt_option_ord_pair_freq',
    'opt_2_option_freq', 'opt_2_tgt_option_pair_freq', 'opt_2_tgt_option_ord_pair_freq',
    'opt_3_option_freq', 'opt_3_tgt_option_pair_freq', 'opt_3_tgt_option_ord_pair_freq'
]


def add_rel_distance_features(df: pd.DataFrame,
                              gt_obj_counts: pd.Series,
                              pair_counts: pd.Series,
                              ord_pair_counts: pd.Series) -> pd.DataFrame:
    """Inject leakage‑free frequency features for rel‑distance questions."""
    df = df.copy()
    for i in range(4):
        df[f"opt_{i}_option_freq"] = df[f"object_{i+1}"].map(gt_obj_counts).fillna(0)
        df[f"opt_{i}_tgt_option_pair_freq"] = df["tgt_gt_pair"].map(pair_counts).fillna(0)
        df[f"opt_{i}_tgt_option_ord_pair_freq"] = df["tgt_gt_ord_pair"].map(ord_pair_counts).fillna(0)

    df["max_option_freq"] = df[[f"opt_{i}_option_freq" for i in range(4)]].max(axis=1)
    df["max_tgt_option_pair_freq"] = df[[f"opt_{i}_tgt_option_pair_freq" for i in range(4)]].max(axis=1)
    df["max_tgt_option_ord_pair_freq"] = df[[f"opt_{i}_tgt_option_ord_pair_freq" for i in range(4)]].max(axis=1)
    return df


def evaluate_rel_distance_cv(df=qdf_rel, n_splits: int = 5, random_state: int = 42, verbose: bool = True):
    """Leakage‑free CV on object_rel_distance bias model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    pbar = tqdm(enumerate(skf.split(df, df["ground_truth"]), 1), total=n_splits, desc="[REL] CV Folds")
    for fold, (train_idx, test_idx) in pbar:
        tr, te = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

        gt_counts = tr['gt_object'].value_counts()
        pair_counts = tr['tgt_gt_pair'].value_counts()
        ord_counts = tr['tgt_gt_ord_pair'].value_counts()

        tr = add_rel_distance_features(tr, gt_counts, pair_counts, ord_counts)
        te = add_rel_distance_features(te, gt_counts, pair_counts, ord_counts)

        X_tr, X_te = tr[rel_feature_cols].copy(), te[rel_feature_cols].copy()
        encode_categoricals(X_tr, X_te)
        y_tr, y_te = tr['ground_truth'], te['ground_truth']

        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        scores.append(acc)
        pbar.set_postfix({"avg_acc": f"{np.mean(scores):.2%}"})

    mean_acc, std_acc = np.mean(scores), np.std(scores)
    if verbose:
        print(f"\n[REL] Overall: {mean_acc:.2%} ± {std_acc:.2%} (n_splits={n_splits})")
        print(f"[REL] Scores for each fold: {[f'{score:.2%}' for score in scores]}")

    # --- Full‑data feature importances (OK to use all rows) ------------------
    full_gt = df['gt_object'].value_counts()
    full_pair = df['tgt_gt_pair'].value_counts()
    full_ord_pair = df['tgt_gt_ord_pair'].value_counts()
    full_df = add_rel_distance_features(df, full_gt, full_pair, full_ord_pair)

    X_full = full_df[rel_feature_cols].copy()
    encode_categoricals(X_full, X_full.copy())
    y_full = full_df['ground_truth']
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_full, y_full)
    fi_rel = (pd.DataFrame({'feature': rel_feature_cols,
                            'importance': model.feature_importances_})
              .sort_values('importance', ascending=False).reset_index(drop=True))
    if verbose:
        print("\n[REL] Feature Importances (from final model trained on all data):")
        print(fi_rel)
    return mean_acc, std_acc, fi_rel

# =============================================================================
# 3.  OBJ‑APPEARANCE‑ORDER BIAS MODEL -----------------------------------------
# =============================================================================

qdf_order = df_full[df_full['question_type'] == 'obj_appearance_order'].copy()

# »–– 3.1  Pre‑extract sequences & helper columns ––«

qdf_order['gt_idx'] = qdf_order['ground_truth'].apply(lambda x: 'ABCD'.index(x))
qdf_order['gt_option'] = qdf_order.apply(lambda r: r['options'][r['gt_idx']].split('. ')[-1], axis=1)

for i in range(4):
    # GT object at each position within GT sequence
    qdf_order[f'gt_obj_{i+1}'] = qdf_order['gt_option'].apply(lambda s: s.split(', ')[i].strip())
    # Parsed option sequences (lists of four obj names) for each answer choice
    qdf_order[f'opt_seq_{i+1}'] = qdf_order['options'].apply(
        lambda opts, idx=i: opts[idx].split('. ', 1)[1].split(', '))

# Feature list ----------------------------------------------------------------
order_feature_cols = [
    *[f'opt_seq_{i}' for i in range(1, 5)],
    *[f'seq_{i}_{comp}score' for i in range(4) for comp in ['pos_', 'pair_', 'comb_pair_', '']],
    # # TODO: everything below leverages privileged info. Remove?
    # 'gt_pos_score', 'gt_pair_score', 'gt_comb_pair_score', 'gt_obj_score',
    # 'max_distractor_pos_score', 'max_distractor_pair_score', 'max_distractor_comb_pair_score',
    # 'max_distractor_score',
    # 'relative_bias_pos_score', 'relative_bias_pair_score', 'relative_bias_comb_pair_score', 'relative_bias_score'
]


def _compute_norm_freq_maps(train_df: pd.DataFrame):
    """Return three dicts of min‑max‑scaled counts from TRAIN ONLY."""
    # Position frequencies ---------------------------------------------------
    pos_counter = Counter()
    for pos in range(1, 5):
        counts = train_df[f'gt_obj_{pos}'].value_counts()
        for obj, c in counts.items():
            pos_counter[(obj, pos)] += c

    # Adjacent pair frequencies ---------------------------------------------
    pair_counter = Counter()
    for pos in range(1, 4):
        pairs = zip(train_df[f'gt_obj_{pos}'], train_df[f'gt_obj_{pos+1}'])
        pair_counter.update(pairs)

    # Combination pair frequencies (i < j) -----------------------------------
    comb_counter = Counter()
    for i in range(1, 5):
        for j in range(i+1, 5):
            pairs = zip(train_df[f'gt_obj_{i}'], train_df[f'gt_obj_{j}'])
            comb_counter.update(pairs)

    # --- Min‑max normalise each counter -------------------------------------
    def _scale(counter):
        if not counter:
            return {}
        arr = np.array(list(counter.values())).reshape(-1, 1)
        scaled = minmax_scale(arr) if len(np.unique(arr)) > 1 else np.ones_like(arr)
        return {k: scaled[i][0] for i, k in enumerate(counter.keys())}

    return _scale(pos_counter), _scale(pair_counter), _scale(comb_counter)


def _add_order_bias_feats(df: pd.DataFrame,
                          norm_pos_map: dict,
                          norm_pair_map: dict,
                          norm_comb_map: dict) -> pd.DataFrame:
    """Add position/pair/comb bias features to **copy** of DF and return it."""
    df = df.copy()
    bias_infos = []
    for _, row in df.iterrows():
        info = {'id': row['id']}
        max_d_pos = max_d_pair = max_d_comb = max_d_score = -np.inf

        for i in range(4):
            seq = row[f'opt_seq_{i+1}']
            # Convert list→string for later label‑encoding
            df.at[row.name, f'opt_seq_{i+1}'] = '|'.join(seq)

            pos_score = pair_score = comb_score = 0.0
            for j, obj in enumerate(seq):
                pos_score += norm_pos_map.get((obj, j+1), 0)
                if j < len(seq)-1:
                    pair_score += norm_pair_map.get((seq[j], seq[j+1]), 0)
                for k in range(j+1, len(seq)):
                    comb_score += norm_comb_map.get((seq[j], seq[k]), 0)
            score = (pos_score + pair_score + comb_score) / 3.0

            info[f'seq_{i}_pos_score'] = pos_score
            info[f'seq_{i}_pair_score'] = pair_score
            info[f'seq_{i}_comb_pair_score'] = comb_score
            info[f'seq_{i}_score'] = score

            if i == row['gt_idx']:
                info['gt_pos_score'] = pos_score
                info['gt_pair_score'] = pair_score
                info['gt_comb_pair_score'] = comb_score
                info['gt_obj_score'] = score
            else:
                max_d_pos = max(max_d_pos, pos_score)
                max_d_pair = max(max_d_pair, pair_score)
                max_d_comb = max(max_d_comb, comb_score)
                max_d_score = max(max_d_score, score)

        info['max_distractor_pos_score'] = max_d_pos
        info['max_distractor_pair_score'] = max_d_pair
        info['max_distractor_comb_pair_score'] = max_d_comb
        info['max_distractor_score'] = max_d_score

        info['relative_bias_pos_score'] = info['gt_pos_score'] - max_d_pos
        info['relative_bias_pair_score'] = info['gt_pair_score'] - max_d_pair
        info['relative_bias_comb_pair_score'] = info['gt_comb_pair_score'] - max_d_comb
        info['relative_bias_score'] = (
            info['relative_bias_pos_score'] +
            info['relative_bias_pair_score'] +
            info['relative_bias_comb_pair_score']
        ) / 3.0
        bias_infos.append(info)

    bias_df = pd.DataFrame(bias_infos)
    return df.merge(bias_df, on='id', how='left')


def evaluate_obj_order_cv(n_splits: int = 5, random_state: int = 42, verbose: bool = True):
    """Leakage‑free CV + feature importances for obj_appearance_order."""
    df = qdf_order  # shorthand
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    pbar = tqdm(enumerate(skf.split(df, df['ground_truth']), 1), total=n_splits, desc="[ORDER] CV Folds")
    for fold, (tr_idx, te_idx) in pbar:
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

        norm_pos_map, norm_pair_map, norm_comb_map = _compute_norm_freq_maps(tr)
        tr = _add_order_bias_feats(tr, norm_pos_map, norm_pair_map, norm_comb_map)
        te = _add_order_bias_feats(te, norm_pos_map, norm_pair_map, norm_comb_map)

        X_tr, X_te = tr[order_feature_cols].copy(), te[order_feature_cols].copy()
        encode_categoricals(X_tr, X_te)
        y_tr, y_te = tr['ground_truth'], te['ground_truth']

        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        scores.append(acc)
        pbar.set_postfix({"avg_acc": f"{np.mean(scores):.2%}"})

    m, s = np.mean(scores), np.std(scores)
    if verbose:
        print(f"\n[ORDER] Overall: {m:.2%} ± {s:.2%} (n_splits={n_splits})")
        print(f"[ORDER] Scores for each fold: {[f'{score:.2%}' for score in scores]}")

    # --- Full‑data importances ---------------------------------------------
    full_norm_maps = _compute_norm_freq_maps(df)
    full_df = _add_order_bias_feats(df.copy(), *full_norm_maps)
    X_full = full_df[order_feature_cols].copy()
    encode_categoricals(X_full, X_full.copy())
    y_full = full_df['ground_truth']

    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_full, y_full)
    fi_order = (pd.DataFrame({'feature': order_feature_cols,
                              'importance': model.feature_importances_})
                .sort_values('importance', ascending=False).reset_index(drop=True))
    if verbose:
        print("\n[ORDER] Feature Importances (from final model trained on all data):")
        print(fi_order)
    return m, s, fi_order

# =============================================================================
# 4.  MAIN --------------------------------------------------------------------
# =============================================================================


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', '-k', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--random_state', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed output')
    args = parser.parse_args()

    n_splits = args.n_splits
    random_state = args.random_state 
    verbose = args.verbose

    print("================  OBJECT REL DISTANCE  ================")
    evaluate_rel_distance_cv(n_splits=n_splits, random_state=random_state, verbose=verbose)

    print("\n================  OBJ APPEARANCE ORDER  ================")
    evaluate_obj_order_cv(n_splits=n_splits, random_state=random_state, verbose=verbose)
