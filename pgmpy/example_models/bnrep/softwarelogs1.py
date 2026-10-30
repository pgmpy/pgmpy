from .._base import BaseExampleModel, BIFMixin


class Softwarelogs1(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/softwarelogs1",
        "n_nodes": 5,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/softwarelogs1.bif"
