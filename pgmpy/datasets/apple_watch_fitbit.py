from pgmpy.datasets._base import _BaseDataset


class AppleWatchFitbit(_BaseDataset):
    """
    References
    ----------
    .. [1] https://www.kaggle.com/aleespinosa/apple-watch-and-fitbit-data
    """

    _tags = {
        "name": "apple_watch_fitbit",
        "n_variables": 18,
        "n_samples": 6264,
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

    base_url = "real/apple-watch-fitbit"

    data_url = "data/aw-fb-pruned18.data.mixed.maximum.6.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/aw-fb-pruned18.knowledge.txt"

    categorical_variables = [
        "gender",
        "device",
        "activity",
    ]
    ordinal_variables = dict()
