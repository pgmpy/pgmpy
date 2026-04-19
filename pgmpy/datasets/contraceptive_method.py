from pgmpy.datasets._base import _BaseDataset


class ContraceptiveMethod(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
    """

    _tags = {
        "name": "contraceptive_method",
        "n_variables": 9,
        "n_samples": 1473,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "real/contraceptive-method"

    data_url = "data/contraceptive-method.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = ["wife-relig", "husb-occ", "media-exp", "contrac-method"]
    ordinal_variables = {
        "wife-educ": [1, 2, 3, 4],
        "husb-educ": [1, 2, 3, 4],
        "sol-index": [1, 2, 3, 4],
    }
