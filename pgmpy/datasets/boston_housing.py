from pgmpy.datasets._base import _BaseDataset


class BostonHousing(_BaseDataset):
    """
    References
    ----------
    - :cite:p:`zhao_hastie_2019`
    - :cite:p:`peerj_wine_quality`
    """

    _tags = {
        "name": "boston_housing",
        "n_variables": 14,
        "n_samples": 506,
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

    base_url = "real/boston-housing"

    data_url = "data/boston-housing.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
