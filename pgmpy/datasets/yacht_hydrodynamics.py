from pgmpy.datasets._base import _BaseDataset


class YachtHydrodynamics(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
    """

    _tags = {
        "name": "yacht_hydrodynamics",
        "n_variables": 7,
        "n_samples": 308,
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

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/yacht-hydrodynamics/"

    data_url = base_url + "data/yacht.hydrodynamics.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/yacht-hydrodynamics.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()
