from .._base import BIFMixin, _BaseExampleModel


class Beam1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/beam1",
        "n_nodes": 6,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/beam1.bif"
