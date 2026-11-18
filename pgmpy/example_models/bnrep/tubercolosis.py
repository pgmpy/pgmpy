from .._base import BaseExampleModel, BIFMixin


class Tubercolosis(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/tubercolosis",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/tubercolosis.bif"
