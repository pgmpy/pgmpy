from pgmpy.datasets._base import _BaseDataset


class DepressionCoping(_BaseDataset):
    _tags = {
        "name": "depression_coping",
        "n_variables": 78,
        "n_samples": 127,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "has_index_col": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/depression-coping/"

    data_url = base_url + "data/depressioncoping.continuous.dat"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "*"

    categorical_variables = []
    ordinal_variables = dict()
