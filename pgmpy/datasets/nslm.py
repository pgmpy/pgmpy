from pgmpy.datasets._base import _BaseDataset


class NSLM(_BaseDataset):
    """
    References
    ----------
    .. [1] Susan Athey, & Stefan Wager. (2019). Estimating Treatment Effects with Causal Forests: An Application.
    .. [2] https://github.com/grf-labs/grf/tree/master/experiments/acic18
    .. [3] https://github.com/pgmpy/example_datasets/tree/main/nslm
    """

    _tags = {
        "name": "nslm",
        "n_variables": 13,
        "n_samples": 10391,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/nslm/"

    data_url = base_url + "data/synthetic_data.csv"
    ground_truth_url = None
    expert_knowledge_url = None

    missing_values_marker = None
    sep = ","

    categorical_variables = ["schoolid", "Z", "C1", "C2", "C3", "XC"]
    ordinal_variables = dict()
