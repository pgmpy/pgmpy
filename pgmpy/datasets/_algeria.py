from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class Algeria(_BaseDataset):
    name = "algeria_forest"
    tags = {
        "n_variables": 15,
        "n_samples": 244,
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
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/"
        "heads/main/real/algerian-forest-fires/"
    )

    data_url = base_url + "data/algerian-forest-fires.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/algerian-forest-fires.knowledge.txt"
