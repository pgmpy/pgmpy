from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class SeoulBike(_BaseDataset):
    name = "seoul_bike"
    tags = {
        "n_variables": 13,
        "n_samples": 8760,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/seoul-bike/"
    )

    data_url = base_url + "data/seoul-bike.mixed.maximum.4.txt"
    ground_truth_url = None
    expert_knowledge_url = None
