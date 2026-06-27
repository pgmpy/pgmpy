from .._base import BaseExampleModel, BIFMixin


class Mountaingoat(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/mountaingoat",
        "n_nodes": 7,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/mountaingoat.bif"
