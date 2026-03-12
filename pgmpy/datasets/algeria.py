from pgmpy.datasets._base import _BaseDataset


class Algeria(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++
    .. [2] https://www.nwcg.gov/publications/pms437/cffdrs/fire-weather-index-system
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

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/"
        "heads/main/real/algerian-forest-fires/"
    )

    data_url = base_url + "data/algerian-forest-fires.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/algerian-forest-fires.knowledge.txt"

    categorical_variables = [
        "Region",
        "day",
        "month",
        "year",
        "Fire",
    ]
    ordinal_variables = dict()
