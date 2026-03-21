from .._base import BIFMixin, _BaseExampleModel


class Rainstorm(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/rainstorm",
        "n_nodes": 34,
        "n_edges": 33,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/rainstorm.bif"
