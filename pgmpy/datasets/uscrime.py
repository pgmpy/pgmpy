from pgmpy.datasets._base import _BaseDataset


class USCrime(_BaseDataset):
    """
    References
    ----------
    .. [1] Der, G., and Everitt, B.S. (2002). A Handbook of Statistical Analysis using SAS, 2e. CRC.
    .. [2] https://www.imsbio.co.jp/RGM/R_rdfile?f=ACSWR/man/usc.Rd&d=R_CC
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

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/uscrime/"

    data_url = base_url + "data/uscrime.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = [
        "S",
    ]
    ordinal_variables = dict()
