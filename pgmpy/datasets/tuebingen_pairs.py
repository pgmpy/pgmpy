from pgmpy.datasets._base import _BaseDataset, _TuebingenBenchmarkMixin


class Tubingen(_TuebingenBenchmarkMixin, _BaseDataset):
    """
    Tübingen Cause-Effect Pairs Dataset.
    A benchmark collection of independent cause-effect pairs.
    """

    _tags = {
        "name": "tubingen",
        "n_variables": None,
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
    base_url = "https://webdav.tuebingen.mpg.de/cause-effect/"
    data_url = "https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip"
