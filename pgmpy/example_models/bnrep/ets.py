from .._base import PlainBIFMixin, _BaseExampleModel


class Ets(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/ets",
        "n_nodes": 20,
        "n_edges": 32,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/ets.bif"
