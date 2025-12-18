from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class MyocardialInfarctionComplications(_BaseDataset):
    name = "myocardial_infarction_complications"

    tags = {
        "n_variables": 124,
        "n_samples": 1700,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/myocardial-infarction-complications/"
    )

    data_url = base_url + "data/myocardial-infarction-complications.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = (
        base_url + "ground.truth/myocardial-infarction-complications.knowledge.txt"
    )
    missing_values_marker = "*"
