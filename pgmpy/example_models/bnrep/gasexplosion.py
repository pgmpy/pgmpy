from .._base import BaseExampleModel, BIFMixin


class Gasexplosion(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/gasexplosion",
        "n_nodes": 18,
        "n_edges": 23,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/gasexplosion.bif"
