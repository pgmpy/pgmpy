from pgmpy.datasets._base import _BaseDataset, _TubingenBenchmarkMixin


class Tubingen(_TubingenBenchmarkMixin, _BaseDataset):
    """
    Tubingen Cause-Effect Pairs Dataset.
    A benchmark collection of independent cause-effect pairs.
    """

    _tags = {
        "name": "tubingen",
        "n_variables": 2,
        "n_samples": None,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": True,
        "is_ordinal": False,
    }
    base_url = "pairwise-tubingen/pairs"
