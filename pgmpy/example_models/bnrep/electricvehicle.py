from .._base import BaseExampleModel, BIFMixin


class Electricvehicle(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/electricvehicle",
        "n_nodes": 23,
        "n_edges": 22,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/electricvehicle.bif"
