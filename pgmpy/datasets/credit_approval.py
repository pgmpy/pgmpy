from pgmpy.datasets._base import _BaseDataset


class CreditApproval(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/Credit+Approval
    """

    _tags = {
        "name": "credit_approval",
        "n_variables": 16,
        "n_samples": 690,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/credit-approval/"

    data_url = base_url + "data/crx.data.mixed.maximum.14.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = [
        "A1",
        "A4",
        "A5",
        "A6",
        "A7",
        "A9",
        "A10",
        "A12",
        "A13",
        "A16",
    ]
    ordinal_variables = dict()
