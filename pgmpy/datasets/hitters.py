from pgmpy.datasets._base import _BaseDataset


class Hitters(_BaseDataset):
    """
    References
    ----------
    - :cite:p:`islr_hitters`
    """

    _tags = {
        "name": "hitters",
        "n_variables": 20,
        "n_samples": 322,
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

    base_url = "real/hitters"

    data_url = "data/hitters.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "*"

    categorical_variables = ["League", "Division", "NewLeague"]
    ordinal_variables = dict()
