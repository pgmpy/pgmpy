from .._base import BIFMixin, _BaseExampleModel


class Earthquake(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/earthquake",
        "n_nodes": 40,
        "n_edges": 77,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/earthquake.bif"
