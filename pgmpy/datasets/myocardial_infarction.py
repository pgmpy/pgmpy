from pgmpy.datasets._base import _BaseDataset


class MyocardialInfarction(_BaseDataset):
    """
    References
    ----------
    .. [1] https://archive.ics.uci.edu/ml/datasets/Myocardial+infarction+complications
    """

    _tags = {
        "name": "myocardial_infarction",
        "n_variables": 123,
        "n_samples": 1700,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": True,
        "has_index_col": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "real/myocardial-infarction-complications"

    data_url = "data/myocarcial-infaraction-complications.continuous.txt"

    ground_truth_url = None
    expert_knowledge_url = "ground.truth/myocarcial-infaraction-complications.knowledge.txt"
    missing_values_marker = "*"

    categorical_variables = []
    ordinal_variables = dict()
