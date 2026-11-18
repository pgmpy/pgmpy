from .._base import BaseExampleModel, BIFMixin


class Drainage(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/drainage",
        "n_nodes": 11,
        "n_edges": 10,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/drainage.bif"
