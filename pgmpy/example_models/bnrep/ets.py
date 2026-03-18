from .._base import BIFMixin, _BaseExampleModel


class Ets(BIFMixin, _BaseExampleModel):
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
