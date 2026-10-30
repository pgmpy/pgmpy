from .._base import BaseExampleModel, BIFMixin


class Algorithms4(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/algorithms4",
        "n_nodes": 4,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/algorithms4.bif"
