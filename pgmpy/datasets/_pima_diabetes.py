from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class PimaDiabetes(_BaseDataset):
    name = "pima_diabetes"

    tags = {
        "n_variables": 9,
        "n_samples": 768,
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
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/pima-diabetes/"
    )

    data_url = base_url + "data/pima-diabetes.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/pima-diabetes.knowledge.txt"
