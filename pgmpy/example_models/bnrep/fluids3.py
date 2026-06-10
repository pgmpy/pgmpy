from .._base import BaseExampleModel, BIFMixin


class Fluids3(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/fluids3",
        "n_nodes": 9,
        "n_edges": 11,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/fluids3.bif"
