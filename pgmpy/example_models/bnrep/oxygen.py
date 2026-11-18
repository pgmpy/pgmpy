from .._base import BaseExampleModel, BIFMixin


class Oxygen(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/oxygen",
        "n_nodes": 31,
        "n_edges": 32,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/oxygen.bif"
