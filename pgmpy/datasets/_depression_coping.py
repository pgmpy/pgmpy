from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class DepressionCoping(_BaseDataset):
    name = "depression_coping"

    tags = {
        "n_variables": 70,  # Based on observation of many columns including STR1-21, DEP1-20, COP1-20, etc.
        "n_samples": 127,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/depression-coping/"
    )

    data_url = base_url + "data/depressioncoping.continuous.dat"
    ground_truth_url = None
    expert_knowledge_url = None
