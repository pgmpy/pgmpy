from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class HungaryChickenpox(_BaseDataset):
    name = "hungary_chickenpox"

    tags = {
        "n_variables": 20,
        "n_samples": 522,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/hungary-chickenpox/"
    )

    data_url = base_url + "data/hungary-chickenpox.continuous.txt"

    ground_truth_url = None
    expert_knowledge_url = None