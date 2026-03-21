from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


class Spartina(_CovarianceMixin, _BaseDataset):
    """
        References
        ----------
        .. [1] Spirtes, P., Glymour, C. N., Scheines, R., & Heckerman, D. (2000). Causation,
    prediction, and search. MIT press, p. 18.
    """

    _tags = {
        "name": "spartina",
        "n_variables": 15,
        "n_samples": 45,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/spartina/"

    data_url = base_url + "data/spartina.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
