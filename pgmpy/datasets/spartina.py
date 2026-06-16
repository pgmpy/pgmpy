from pgmpy.datasets._base import BaseCovarianceDataset


class Spartina(BaseCovarianceDataset):
    """
    References
    ----------
    - :cite:p:`spirtes_glymour_scheines_2001`
    """

    _tags = {
        "name": "spartina",
        "n_variables": 15,
        "n_samples": 45,
        "is_continuous": True,
    }

    base_url = "real/spartina"

    data_url = "data/spartina.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
