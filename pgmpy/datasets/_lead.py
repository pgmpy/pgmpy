from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


@register_dataset_class
class Lead(_CovarianceMixin, _BaseDataset):
    name = "lead"
    tags = {
        "n_variables": 7,
        "n_samples": 221,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/lead/"
    )

    data_url = base_url + "data/lead.cov.txt"

    ground_truth_url = None
    expert_knowledge_url = None
