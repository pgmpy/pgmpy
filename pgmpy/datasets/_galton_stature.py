from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class GaltonStature(_BaseDataset):
    name = "galton_stature"
    tags = {
        "n_variables": 5,
        "n_samples": 898,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/galton-stature/"
    )

    data_url = base_url + "data/galton-stature.mixed.txt"
    ground_truth_url = None
    expert_knowledge_url = None
