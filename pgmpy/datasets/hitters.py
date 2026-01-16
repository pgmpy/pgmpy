from pgmpy.datasets._base import _BaseDataset


class Hitters(_BaseDataset):
    _tags = {
        "name": "hitters",
        "n_variables": 20,
        "n_samples": 322,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/hitters/"
    )

    data_url = base_url + "data/hitters.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "*"
