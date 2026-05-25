from pgmpy.datasets._base import _BaseDataset, _CovarianceMixin


class Lead(_CovarianceMixin, _BaseDataset):
    _tags = {
        "name": "lead",
        "n_variables": 7,
        "n_samples": 221,
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

    base_url = "real/lead"

    data_url = "data/lead.cov.txt"

    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
