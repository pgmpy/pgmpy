from .._base import BaseExampleModel, BIFMixin


class Ecosystem(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/ecosystem",
        "n_nodes": 13,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/ecosystem.bif"
