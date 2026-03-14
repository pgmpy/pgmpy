from .._base import BIFMixin, _BaseExampleModel


class Nuisancegrowth(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/nuisancegrowth",
        "n_nodes": 19,
        "n_edges": 24,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/nuisancegrowth.bif"
