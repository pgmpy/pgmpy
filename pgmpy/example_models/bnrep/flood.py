from .._base import PlainBIFMixin, _BaseExampleModel


class Flood(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/flood",
        "n_nodes": 22,
        "n_edges": 42,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/flood.bif"
