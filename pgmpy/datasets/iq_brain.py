from pgmpy.datasets._base import _BaseDataset


class IQBrainSize(_BaseDataset):
    """
    References
    ----------
    .. [1] http://lib.stat.cmu.edu/datasets/IQ_Brain_Size
    """

    _tags = {
        "name": "iq_brain_size",
        "n_variables": 9,
        "n_samples": 20,
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

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/iq-brain-size/"

    data_url = base_url + "data/iq_brain_size.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = []
    ordinal_variables = dict()
