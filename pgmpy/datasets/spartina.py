from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


class Spartina(_CovarianceMixin, _BaseDataset):
    """
    References
    ----------
    - :cite:p:`spirtes_glymour_scheines_2001`
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

    base_url = "real/spartina"

    data_url = "data/spartina.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
