from pgmpy.datasets._base import _BaseDataset


class PimaDiabetes(_BaseDataset):
    """
    References
    ----------
    .. [1] https://www.kaggle.com/uciml/pima-indians-diabetes-database
    """

    _tags = {
        "name": "pima_diabetes",
        "n_variables": 9,
        "n_samples": 768,
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

    base_url = "real/pima-diabetes"

    data_url = "data/pima-diabetes.mixed.maximum.2.txt"
    ground_truth_url = None
    expert_knowledge_url = "ground.truth/pima-diabetes.knowledge.txt"

    categorical_variables = ["Outcome"]
    ordinal_variables = dict()
