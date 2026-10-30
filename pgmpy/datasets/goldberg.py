from pgmpy.datasets._base import BaseCovarianceDataset


class Goldberg(BaseCovarianceDataset):
    _tags = {
        "name": "goldberg",
        "n_variables": 6,
        "n_samples": 645,
        "is_continuous": True,
    }

    base_url = "real/goldberg"

    data_url = "data/goldberg.cov.txt"
    ground_truth_url = None
    expert_knowledge_url = None
