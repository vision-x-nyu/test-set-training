# from .models import MMMUMCModel
from .models import MMMUModelSubset, MMMUMCModel
from .data_loader import load_data


# TODO: make a get_models for LLMs?
def get_models():
    """Get all MMMU benchmark models (just the global model for now)."""
    # # MC model
    return [MMMUMCModel()]

    # subfield models
    df = load_data()
    # key = "subfield_cluster" # 30%
    # key = "topic_difficulty" # 30%
    key = "subfield"  # 34%
    subsets = df[key].unique()
    models = [MMMUModelSubset(key=key, val=subset) for subset in subsets]
    return models


__all__ = ["load_data", "get_models"]
