from .._base import BaseExampleModel, BIFMixin


class Blacksea(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/blacksea",
        "n_nodes": 48,
        "n_edges": 87,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/blacksea.bif"
