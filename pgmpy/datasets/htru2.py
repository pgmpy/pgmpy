from pgmpy.datasets._base import _BaseDataset


class HTRU2(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/HTRU2
    """

    _tags = {
        "name": "htru2",
        "n_variables": 9,
        "n_samples": 17898,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/"
        "real/htru2/"
    )

    data_url = base_url + "data/pulsar.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/htr2.knowledge.txt"

    categorical_variables = ["pulsar"]
    ordinal_variables = dict()
