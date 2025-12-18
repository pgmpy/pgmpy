from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class CollegePlans(_BaseDataset):
    name = "college_plans"

    tags = {
        "n_variables": 5,
        "n_samples": 10318,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": True,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/college-plans/"
    )

    data_url = base_url + "data/college-plans.discrete.txt"
    ground_truth_url = None
    expert_knowledge_url = None
