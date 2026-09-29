from .._base import BaseExampleModel, BIFMixin


class Gasifier(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/gasifier",
        "n_nodes": 40,
        "n_edges": 39,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/gasifier.bif"
