from pgmpy.datasets._base import BaseDataset


class TwinsDataset(BaseDataset):
    _tags = {
        "name": "twins",
        "n_variables": 56,
        "n_samples": 71345,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "twins"

    data_url = "twins.txt"

    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = []
    ordinal_variables = dict()
