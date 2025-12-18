from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


@register_dataset_class
class Spartina(_CovarianceMixin, _BaseDataset):
    name = "spartina"
    tags = {
        "n_variables": 15,
        "n_samples": 45,
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
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/spartina/"
    )

    data_url = base_url + "data/spartina.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = None
