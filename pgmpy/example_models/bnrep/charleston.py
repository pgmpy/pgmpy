from .._base import BaseExampleModel, BIFMixin


class Charleston(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/charleston",
        "n_nodes": 24,
        "n_edges": 35,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/charleston.bif"
