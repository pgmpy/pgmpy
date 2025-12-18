from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class AbaloneContinuous(_BaseDataset):
    name = "abalone_continuous"
    tags = {
        "n_variables": 8,
        "n_samples": 4177,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/abalone/"

    data_url = base_url + "data/abalone.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/abalone.knowledge.txt"


@register_dataset_class
class AbaloneMixed(_BaseDataset):
    name = "abalone_mixed"
    tags = {
        "n_variables": 9,
        "n_samples": 4177,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/abalone/"

    data_url = base_url + "data/abalone.mixed.maximum.3.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/abalone.knowledge.txt"
