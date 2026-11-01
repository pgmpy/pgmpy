from .._base import BaseExampleModel, BIFMixin


class Covid1(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/covid1",
        "n_nodes": 12,
        "n_edges": 11,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/covid1.bif"
