from .._base import BaseExampleModel, BIFMixin


class Aerialvehicles(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/aerialvehicles",
        "n_nodes": 39,
        "n_edges": 41,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/aerialvehicles.bif"
