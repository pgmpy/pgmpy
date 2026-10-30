from .._base import BaseExampleModel, BIFMixin


class Realestate1(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/realestate1",
        "n_nodes": 27,
        "n_edges": 77,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/realestate1.bif"
