from .._base import BaseExampleModel, BIFMixin


class Tastingtea(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/tastingtea",
        "n_nodes": 17,
        "n_edges": 72,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/tastingtea.bif"
