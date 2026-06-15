from .._base import BaseExampleModel, BIFMixin


class Electrolysis(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/electrolysis",
        "n_nodes": 16,
        "n_edges": 18,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/electrolysis.bif"
