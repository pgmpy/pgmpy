from .._base import BaseExampleModel, BIFMixin


class Concrete4(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/concrete4",
        "n_nodes": 4,
        "n_edges": 3,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/concrete4.bif"
