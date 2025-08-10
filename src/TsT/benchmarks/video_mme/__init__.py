from .models import VideoMMEModel, VideoMMEModelSubset, VideoMMEModelSubsetCombo
from .data_loader import load_data


# ── constants ──────────────────────────────────────────────────────────────────
# 3
durations = ["short", "medium", "long"]
# 6
domains = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual",
]
# 12
task_types = [
    "Counting Problem",
    "Information Synopsis",
    "Object Recognition",
    "Action Reasoning",
    "Object Reasoning",
    "Temporal Perception",
    "Attribute Perception",
    "Temporal Reasoning",
    "Action Recognition",
    "OCR Problems",
    "Spatial Perception",
    "Spatial Reasoning",
]
# 30
sub_cats = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]


# ── global models ──────────────────────────────────────────────────────────────
global_models = [
    VideoMMEModel(),
]

# ── sub-models ───────────────────────────────────────────────────────────────
task_models = [VideoMMEModelSubset(key="task_type", val=task) for task in task_types]
domain_models = [VideoMMEModelSubset(key="domain", val=domain) for domain in domains]
sub_cat_models = [VideoMMEModelSubset(key="sub_category", val=sub_cat) for sub_cat in sub_cats]
duration_models = [VideoMMEModelSubset(key="duration", val=duration) for duration in durations]

# ── combo-sub-models ──────────────────────────────────────────────────────────
# 3 * 6 * 12 * 30 = 6480
all_combos = [
    VideoMMEModelSubsetCombo(
        key_vals={"duration": duration, "domain": domain, "sub_category": sub_cat, "task_type": task}
    )
    for duration in durations
    for domain in domains
    for task in task_types
    for sub_cat in sub_cats
]
# 12 * 30 = 360
task_subcat_combos = [
    VideoMMEModelSubsetCombo(key_vals={"task_type": task, "sub_category": sub_cat})
    for task in task_types
    for sub_cat in sub_cats
]
# 6 * 12 = 72
domain_task_combos = [
    VideoMMEModelSubsetCombo(key_vals={"domain": domain, "task_type": task})
    for domain in domains
    for task in task_types
]
# 3 * 12 = 36
duration_task_combos = [
    VideoMMEModelSubsetCombo(key_vals={"duration": duration, "task_type": task})
    for duration in durations
    for task in task_types
]
# 3 * 6 = 18
duration_domain_combos = [
    VideoMMEModelSubsetCombo(key_vals={"duration": duration, "domain": domain})
    for duration in durations
    for domain in domains
]
# 3 * 30 = 90
duration_subcat_combos = [
    VideoMMEModelSubsetCombo(key_vals={"duration": duration, "sub_category": sub_cat})
    for duration in durations
    for sub_cat in sub_cats
]


def get_models():
    """Get all Video-MME benchmark models."""

    # ── global models ──────────────────────────────────────────────────────────
    return global_models  # random

    # ── sub-models ────────────────────────────────────────────────────────────
    # return task_models  # ok
    # return duration_models  # random
    # return domain_models  # random
    # return sub_cat_models  # random

    # ── combo-sub-models ────────────────────────────────────────────────────────
    # return all_combos
    # return task_subcat_combos
    # return domain_task_combos
    # return duration_task_combos
    # return duration_domain_combos
    # return duration_subcat_combos


__all__ = ["load_data", "get_models"]
