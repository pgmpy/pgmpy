from pgmpy.datasets._base import _BaseDataset


class PittsburghBridges(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/Pittsburgh+Bridges
    """

    _tags = {
        "name": "pittsburgh_bridges",
        "n_variables": 12,
        "n_samples": 108,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "real/pittsburgh-bridges"

    data_url = "data/bridges.data.version21.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "?"

    categorical_variables = [
        "RIVER",
        "LOCATION",
        "ERECTED",
        "PURPOSE",
        "LENGTH",
        "LANES",
        "CLEAR-G",
        "T-OR-D",
        "MATERIAL",
        "SPAN",
        "REL-L",
        "TYPE",
    ]
    ordinal_variables = dict()
