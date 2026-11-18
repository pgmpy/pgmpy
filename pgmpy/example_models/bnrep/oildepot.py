from .._base import BaseExampleModel, BIFMixin


class Oildepot(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/oildepot",
        "n_nodes": 41,
        "n_edges": 40,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/oildepot.bif"
