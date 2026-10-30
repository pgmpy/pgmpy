from .._base import BaseExampleModel, BIFMixin


class Algalactivity2(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/algalactivity2",
        "n_nodes": 8,
        "n_edges": 15,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/algalactivity2.bif"
