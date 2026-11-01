from .._base import BaseExampleModel, BIFMixin


class Bank(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/bank",
        "n_nodes": 4,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/bank.bif"
