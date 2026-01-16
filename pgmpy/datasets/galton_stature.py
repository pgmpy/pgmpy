from pgmpy.datasets._base import _BaseDataset


class GaltonStature(_BaseDataset):
    """
    References
    ----------
    .. [1] http://www.medicine.mcgill.ca/epidemiology/hanley/galton/
    """

    _tags = {
        "name": "galton_stature",
        "n_variables": 5,
        "n_samples": 898,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/galton-stature/"
    )

    data_url = base_url + "data/galton-stature.mixed.txt"
    ground_truth_url = None
    expert_knowledge_url = None

    categorical_variables = ["family", "Gender"]
    ordinal_variables = dict()
