from .._base import PlainBIFMixin, _BaseExampleModel


class Rainwater(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/rainwater",
        "n_nodes": 6,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/rainwater.bif"
