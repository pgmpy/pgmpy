from .._base import BaseExampleModel, BIFMixin


class Redmeat(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/redmeat",
        "n_nodes": 11,
        "n_edges": 11,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/redmeat.bif"
