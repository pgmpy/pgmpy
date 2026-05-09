from pgmpy.datasets._base import _BaseDataset


class Algeria(_BaseDataset):
    """
    References
    ----------
    - :cite:p:`uci_algerian_forest_fires`
    - :cite:p:`peerj_pima`
    """

    _tags = {
        "name": "algerian_forest",
        "n_variables": 15,
        "n_samples": 244,
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

    base_url = "real/algerian-forest-fires"

    data_url = "data/algerian-forest-fires.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/algerian-forest-fires.knowledge.txt"

    categorical_variables = [
        "Region",
        "day",
        "month",
        "year",
        "Fire",
    ]
    ordinal_variables = dict()
