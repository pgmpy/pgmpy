from .._base import BaseExampleModel, BIFMixin


class Transport(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/transport",
        "n_nodes": 6,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/transport.bif"
