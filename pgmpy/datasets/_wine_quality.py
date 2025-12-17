from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset

BASE_URL = (
    "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
    "real/wine-quality/"
)

EXPERT_URL = BASE_URL + "ground.truth/wine.quality.knowledge.txt"


@register_dataset_class
class WineQualityRed(_BaseDataset):
    name = "wine_quality_red"

    tags = {
        "n_variables": 12,
        "n_samples": 1599,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = BASE_URL
    data_url = BASE_URL + "data/winequality-red.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = EXPERT_URL


@register_dataset_class
class WineQualityWhite(_BaseDataset):
    name = "wine_quality_white"

    tags = {
        "n_variables": 12,
        "n_samples": 4898,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = BASE_URL
    data_url = BASE_URL + "data/winequality-white.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = EXPERT_URL


@register_dataset_class
class WineQualityRedWhiteMixed(_BaseDataset):
    name = "wine_quality_red_white_mixed"

    tags = {
        "n_variables": 13,
        "n_samples": 6497,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = BASE_URL
    data_url = BASE_URL + "data/winequality-red-white.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = EXPERT_URL
