from pgmpy.datasets._base import BaseDataset


class AngristKrueger(BaseDataset):
    """
    References
    ----------
    - :cite:p:`angrist_krueger_1991`
    - :cite:p:`angrist_krueger_qob_dataset`
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

    base_url = "real/angrist-krueger-qob"

    data_url = "data/angrist-krueger-qob.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
