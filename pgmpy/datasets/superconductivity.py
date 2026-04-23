from pgmpy.datasets._base import _BaseDataset


class Superconductivity(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/superconductivty+data
    """

    _tags = {
        "name": "superconductivity",
        "n_variables": 82,
        "n_samples": 21263,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "real/superconductivity"
    data_url = "data/superconductivity.continuous.txt"

    ground_truth_url = None
    expert_knowledge_url = "ground.truth/superconductivity.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()
