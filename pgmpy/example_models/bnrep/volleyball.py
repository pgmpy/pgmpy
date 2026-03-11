from .._base import BIFMixin, _BaseExampleModel


class Volleyball(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/volleyball",
        "n_nodes": 14,
        "n_edges": 18,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/volleyball.bif"
