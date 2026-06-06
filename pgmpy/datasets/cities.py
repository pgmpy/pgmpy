from pgmpy.datasets._base import BaseDataset, _CovarianceMixin


class Cities(_CovarianceMixin, BaseDataset):
    """
    References
    ----------
    - :cite:p:`spirtes_glymour_scheines_2001`
    """

    _tags = {
        "name": "cities",
        "n_variables": 7,
        "n_samples": 164,
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

    base_url = "real/cites"

    data_url = "data/cites.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/cites.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()
