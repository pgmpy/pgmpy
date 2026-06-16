from .._base import BaseExampleModel, BIFMixin


class Curacao5(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/curacao5",
        "n_nodes": 13,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/curacao5.bif"
