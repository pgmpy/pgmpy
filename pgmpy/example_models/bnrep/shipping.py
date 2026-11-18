from .._base import BaseExampleModel, BIFMixin


class Shipping(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/shipping",
        "n_nodes": 36,
        "n_edges": 35,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/shipping.bif"
