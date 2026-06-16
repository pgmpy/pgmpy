from pgmpy.datasets._base import BaseDataset


class HungaryChickenpox(BaseDataset):
    """
    References
    ----------
    - :cite:p:`rozemberczki_2021`
    - :cite:p:`uci_hungarian_chickenpox`
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

    base_url = "real/hungary-chickenpox"

    data_url = "data/hungary-chickenpox.continuous.txt"

    ground_truth_url = "ground.truth/hungary_dag.txt"
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
