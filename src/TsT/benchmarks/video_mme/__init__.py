from .models import VideoMMEModel, VideoMMEModelSubset, VideoMMEModelSubsetCombo
from .data_loader import load_data

global_models = [
    VideoMMEModel(),
]

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

task_models = [VideoMMEModelSubset(key="task_type", val=task) for task in task_types]

durations = ["short", "medium", "long"]
duration_models = [VideoMMEModelSubset(key="duration", val=duration) for duration in durations]

domains = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual",
]
domain_models = [VideoMMEModelSubset(key="domain", val=domain) for domain in domains]

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
sub_cat_models = [VideoMMEModelSubset(key="sub_category", val=sub_cat) for sub_cat in sub_cats]


all_combos = [
    VideoMMEModelSubsetCombo(
        key_vals={"domain": domain, "duration": duration, "sub_category": sub_cat, "task_type": task}
    )
    for domain in domains
    for duration in durations
    for sub_cat in sub_cats
    for task in task_types
]

task_subcat_combos = [
    VideoMMEModelSubsetCombo(key_vals={"task_type": task, "sub_category": sub_cat})
    for task in task_types
    for sub_cat in sub_cats
]

task_domain_combos = [
    VideoMMEModelSubsetCombo(key_vals={"task_type": task, "domain": domain})
    for task in task_types
    for domain in domains
]


def get_models():
    """Get all Video-MME benchmark models."""

    # ── global models ──────────────────────────────────────────────────────────
    return global_models

    # ── sub-models ────────────────────────────────────────────────────────────
    # return task_models
    # return duration_models
    # return domain_models
    # return sub_cat_models

    # ── combo-sub-models ────────────────────────────────────────────────────────
    # return all_combos
    # return task_subcat_combos
    # return task_domain_combos


__all__ = ["load_data", "get_models"]
