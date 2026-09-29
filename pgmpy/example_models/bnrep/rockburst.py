from .._base import BaseExampleModel, BIFMixin


class Rockburst(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/rockburst",
        "n_nodes": 6,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/rockburst.bif"
