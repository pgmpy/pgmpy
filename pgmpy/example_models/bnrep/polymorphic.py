from .._base import BaseExampleModel, BIFMixin


class Polymorphic(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/polymorphic",
        "n_nodes": 22,
        "n_edges": 21,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/polymorphic.bif"
