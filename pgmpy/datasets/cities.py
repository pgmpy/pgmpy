from pgmpy.datasets._base import BaseCovarianceDataset


class Cities(BaseCovarianceDataset):
    """
    References
    ----------
    - :cite:p:`spirtes_glymour_scheines_2001`
    """

    _tags = {
        "name": "cities",
        "n_variables": 7,
        "n_samples": 164,
        "has_expert_knowledge": True,
        "is_simulated": True,
        "is_continuous": True,
    }

    base_url = "real/cites"

    data_url = "data/cites.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/cites.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()
