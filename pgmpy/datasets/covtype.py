from pgmpy.datasets._base import _BaseDataset


class CoverType(_BaseDataset):
    _tags = {
        "name": "cover_type",
        "n_variables": 11,
        "n_samples": 581012,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/covertype/"

    data_url = base_url + "data/covtype.11vars.mixed.maximum.7.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/covertype.knowledge.txt"
    categorical_variables = ["Type"]
    ordinal_variables = dict()
