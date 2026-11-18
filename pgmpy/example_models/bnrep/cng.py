from .._base import BaseExampleModel, BIFMixin


class Cng(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/cng",
        "n_nodes": 86,
        "n_edges": 85,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/cng.bif"
