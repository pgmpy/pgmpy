from pgmpy.datasets._base import _BaseDataset


class BlueDriver(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/adult
    """

    _tags = {
        "name": "blue_driver",
        "n_variables": 10,
        "n_samples": 1381,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/blue-driver/"

    data_url = base_url + "data/bluedata2.edited.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = []
    ordinal_variables = {}
