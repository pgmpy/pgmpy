from pgmpy.datasets._base import _BaseDataset

BASE_URL = "real/wine-quality"
EXPERT_URL = "ground.truth/wine.quality.knowledge.txt"


class WineQualityRed(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/wine+quality
    """

    _tags = {
        "name": "wine_quality_red",
        "n_variables": 12,
        "n_samples": 1599,
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

    base_url = BASE_URL
    data_url = "data/winequality-red.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = EXPERT_URL

    categorical_variables = []
    ordinal_variables = dict()


class WineQualityWhite(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/wine+quality
    """

    _tags = {
        "name": "wine_quality_white",
        "n_variables": 12,
        "n_samples": 4898,
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

    base_url = BASE_URL
    data_url = "data/winequality-white.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = EXPERT_URL

    categorical_variables = []
    ordinal_variables = dict()


class WineQualityRedWhiteMixed(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/wine+quality
    """

    _tags = {
        "name": "wine_quality_red_white_mixed",
        "n_variables": 13,
        "n_samples": 6497,
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

    base_url = BASE_URL
    data_url = "data/winequality-red-white.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = EXPERT_URL

    categorical_variables = ["type"]
    ordinal_variables = dict()
