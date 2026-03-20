from pgmpy.datasets._base import _BaseDataset


class Hitters(_BaseDataset):
    """
    References
    ----------
    .. [1] https://gist.githubusercontent.com/keeganhines/59974f1ebef97bbaa44fb19143f90bad/raw
           /d9bcf657f97201394a59fffd801c44347eb7e28d/Hitters.csv
    """

    _tags = {
        "name": "hitters",
        "n_variables": 20,
        "n_samples": 322,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/hitters/"

    data_url = base_url + "data/hitters.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "*"

    categorical_variables = ["League", "Division", "NewLeague"]
    ordinal_variables = dict()
