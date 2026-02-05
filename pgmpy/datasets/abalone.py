from pgmpy.datasets._base import _BaseDataset


class AbaloneContinuous(_BaseDataset):
    """
    References
    ----------
    .. [1] Lopez-Paz, D., Muandet, K., Schölkopf, B., & Tolstikhin, I. (2015, June). Towards a learning theory of
           cause-effect inference. In International Conference on Machine Learning (pp. 1452-1461). PMLR.
    .. [2] https://archive.ics.uci.edu/ml/datasets/abalone
    """

    _tags = {
        "name": "abalone_continuous",
        "n_variables": 8,
        "n_samples": 4177,
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

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/abalone/"

    data_url = base_url + "data/abalone.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/abalone.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()


class AbaloneMixed(_BaseDataset):
    """
    References
    ----------
    .. [1] Lopez-Paz, D., Muandet, K., Schölkopf, B., & Tolstikhin, I. (2015, June). Towards a learning theory of
           cause-effect inference. In International Conference on Machine Learning (pp. 1452-1461). PMLR.
    .. [2] https://archive.ics.uci.edu/ml/datasets/abalone
    """

    _tags = {
        "name": "abalone_mixed",
        "n_variables": 9,
        "n_samples": 4177,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/abalone/"

    data_url = base_url + "data/abalone.mixed.maximum.3.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/abalone.knowledge.txt"

    categorical_variables = [
        "Sex",
    ]
    ordinal_variables = dict()
