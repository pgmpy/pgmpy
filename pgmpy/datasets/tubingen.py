from pgmpy.datasets._base import BaseTubingenDataset


class Tubingen(BaseTubingenDataset):
    """
    Tubingen Cause-Effect Pairs Dataset.
    A benchmark collection of independent cause-effect pairs.
    """

    _tags = {
        "name": "tubingen",
        "n_variables": 2,
        "has_ground_truth": True,
        "is_continuous": True,
        "is_mixed": True,
    }
    base_url = "pairwise-tubingen/pairs"
