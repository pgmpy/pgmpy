from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


class Goldberg(_CovarianceMixin, _BaseDataset):
    _tags = {
        "name": "goldberg",
        "n_variables": 6,
        "n_samples": 645,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/goldberg/"

    data_url = base_url + "data/goldberg.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = None
