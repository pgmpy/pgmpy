from .._base import BaseExampleModel, BIFMixin


class Humanitarian(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/humanitarian",
        "n_nodes": 21,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/humanitarian.bif"
