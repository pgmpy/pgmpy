from pgmpy.datasets._base import BaseDataset


class USCrime(BaseDataset):
    """
    References
    ----------
    - :cite:p:`der_everitt_2002`
    - :cite:p:`acswr_usc_dataset`
    """

    _tags = {
        "name": "uscrime",
        "n_variables": 14,
        "n_samples": 47,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "real/uscrime"

    data_url = "data/uscrime.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = [
        "S",
    ]
    ordinal_variables = dict()
