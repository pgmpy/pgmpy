from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class PittsburghBridges(_BaseDataset):
    name = "pittsburgh_bridges"

    tags = {
        "n_variables": 12,
        "n_samples": 108,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/pittsburgh-bridges/"
    )

    data_url = base_url + "data/bridges.data.version21.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "?"
