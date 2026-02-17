from .._base import PlainBIFMixin, _BaseExampleModel


class Shipping(PlainBIFMixin, _BaseExampleModel):
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
