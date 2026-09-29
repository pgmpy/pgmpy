from .._base import BaseExampleModel, BIFMixin


class Disputed2(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/disputed2",
        "n_nodes": 17,
        "n_edges": 19,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/disputed2.bif"
