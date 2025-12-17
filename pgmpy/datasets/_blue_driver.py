from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class BlueDriver(_BaseDataset):
    name = "blue_driver"
    tags = {
        "n_variables": 10,
        "n_samples": 282,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/blue-driver/"

    data_url = base_url + "data/blue.driver1.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
