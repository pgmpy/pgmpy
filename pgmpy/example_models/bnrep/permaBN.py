from .._base import BaseExampleModel, BIFMixin


class PermaBN(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/permaBN",
        "n_nodes": 14,
        "n_edges": 27,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/permaBN.bif"
