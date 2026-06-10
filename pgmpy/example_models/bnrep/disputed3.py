from .._base import BaseExampleModel, BIFMixin


class Disputed3(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/disputed3",
        "n_nodes": 27,
        "n_edges": 34,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/disputed3.bif"
