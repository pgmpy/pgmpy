from .._base import BIFMixin, _BaseExampleModel


class Yangtze(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/yangtze",
        "n_nodes": 31,
        "n_edges": 54,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/yangtze.bif"
