from pgmpy.datasets._base import _BaseDataset


class DryBean(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
    """

    _tags = {
        "name": "dry_bean",
        "n_variables": 17,
        "n_samples": 13611,
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

    base_url = "real/dry-bean"

    data_url = "data/drybean.data.mixed.maximum.7.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/dry-bean.knowledge.txt"

    categorical_variables = ["Class"]
    ordinal_variables = dict()
