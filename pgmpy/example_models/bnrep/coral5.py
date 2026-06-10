from .._base import BaseExampleModel, BIFMixin


class Coral5(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/coral5",
        "n_nodes": 8,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/coral5.bif"
