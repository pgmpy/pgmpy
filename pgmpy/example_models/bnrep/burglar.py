from .._base import BaseExampleModel, BIFMixin


class Burglar(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/burglar",
        "n_nodes": 3,
        "n_edges": 2,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/burglar.bif"
