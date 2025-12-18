from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class Adult(_BaseDataset):
    name = "adult"
    tags = {
        "n_variables": 15,
        "n_samples": 32561,
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

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/adult/"

    data_url = base_url + "data/adult.data.mixed.maximum.50.json.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/adult.knowledge.txt"
