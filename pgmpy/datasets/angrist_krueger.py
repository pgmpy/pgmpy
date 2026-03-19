from pgmpy.datasets._base import _BaseDataset


class AngristKrueger(_BaseDataset):
    """
    References
    ----------
    .. [1] Angrist, J. D., & Krueger, A. B. (1991). Does Compulsory School Attendance Affect
           Schooling and Earnings? The Quarterly Journal of Economics, 106(4), 979-1014.
    .. [2] https://economics.mit.edu/sites/default/files/publications/asciiqob.zip
    """

    _tags = {
        "name": "angrist_krueger_qob",
        "n_variables": 5,
        "n_samples": 329509,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/angrist-krueger-qob/"

    data_url = base_url + "data/angrist-krueger-qob.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
