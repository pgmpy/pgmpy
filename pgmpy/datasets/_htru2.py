from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class HTRU2(_BaseDataset):
    name = "htru2"

    tags = {
        "n_variables": 9,
        "n_samples": 17898,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/htru2/"
    )

    data_url = base_url + "data/pulsar.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/htr2.knowledge.txt"
