from pgmpy.datasets._base import BaseCovarianceDataset


class Lead(BaseCovarianceDataset):
    _tags = {
        "name": "lead",
        "n_variables": 7,
        "n_samples": 221,
        "is_continuous": True,
    }

    base_url = "real/lead"

    data_url = "data/lead.cov.txt"

    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
