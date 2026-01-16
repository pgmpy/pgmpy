from pgmpy.datasets._base import _BaseDataset


class AutoMpg(_BaseDataset):
    """
    References
    ----------
    .. [1] Lopez-Paz, D., Muandet, K., Schölkopf, B., & Tolstikhin, I. (2015, June). Towards a learning theory of
           cause-effect inference. In International Conference on Machine Learning (pp. 1452-1461). PMLR.
    .. [2] https://archive.ics.uci.edu/ml/datasets/auto+mpg
    """

    _tags = {
        "name": "auto_mpg",
        "n_variables": 8,
        "n_samples": 392,
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

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/auto-mpg/"

    data_url = base_url + "data/auto-mpg.data.mixed.max.3.categories.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/auto-mpg.knowledge.txt"

    categorical_variables = ["cylinders", "modelyear", "origin"]
    ordinal_variables = dict()
