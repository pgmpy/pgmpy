from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class BostonHousing(_BaseDataset):
    name = "boston_housing"

    tags = {
        "n_variables": 14,
        "n_samples": 506,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/boston-housing/"

    data_url = base_url + "data/boston-housing.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
