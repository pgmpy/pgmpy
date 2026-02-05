from pgmpy.datasets._base import _BaseDataset


class HungaryChickenpox(_BaseDataset):
    """
    References
    ----------
    .. [1] Rozemberczki, B., Scherer, P., Kiss, O., Sarkar, R., & Ferenci, T. (2021). Chickenpox cases in hungary:
           a benchmark dataset for spatiotemporal signal processing with graph neural networks.
           arXiv preprint arXiv:2102.08100.
    .. [2] https://archive.ics.uci.edu/ml/datasets/Hungarian+Chickenpox+Cases
    """

    _tags = {
        "name": "hungary_chickenpox",
        "n_variables": 20,
        "n_samples": 522,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/"
        "real/hungary-chickenpox/"
    )

    data_url = base_url + "data/hungary-chickenpox.continuous.txt"

    ground_truth_url = base_url + "ground.truth/hungary_dag.txt"
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
