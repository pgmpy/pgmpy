from .._base import PlainBIFMixin, _BaseExampleModel


class Inverters(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/inverters",
        "n_nodes": 29,
        "n_edges": 33,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/inverters.bif"
