from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class ResidentialBuilding(_BaseDataset):
    name = "residential_building"

    tags = {
        "n_variables": 109,
        "n_samples": 372,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/residential-building/"
    )

    data_url = base_url + "data/residential-building.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
