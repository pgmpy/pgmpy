from pgmpy.datasets._base import _BaseDataset


class BostonHousing(_BaseDataset):
    """
    References
    ----------
    .. [1] Zhao, Q., & Hastie, T. (2021). Causal Interpretations of Black-Box Models. Journal of Business &amp; Economic
           Statistics, 39(1), 272–281. https://doi.org/10.1080/07350015.2019.1624293
    .. [2] https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    """

    _tags = {
        "name": "boston_housing",
        "n_variables": 14,
        "n_samples": 506,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/boston-housing/"

    data_url = base_url + "data/boston-housing.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
