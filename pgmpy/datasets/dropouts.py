from pgmpy.datasets._base import BaseCovarianceDataset


class Dropouts(BaseCovarianceDataset):
    _tags = {
        "name": "dropouts",
        "n_variables": 8,
        "n_samples": 159,
        "is_continuous": True,
    }

    base_url = "real/dropouts"

    data_url = "data/dropouts.cov.txt"

    ground_truth_url = None

    expert_knowledge_url = None
