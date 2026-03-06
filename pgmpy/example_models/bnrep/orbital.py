from .._base import BIFMixin, _BaseExampleModel


class Orbital(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/orbital",
        "n_nodes": 60,
        "n_edges": 59,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/orbital.bif"
