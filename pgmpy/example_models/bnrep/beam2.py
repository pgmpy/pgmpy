from .._base import BIFMixin, _BaseExampleModel


class Beam2(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/beam2",
        "n_nodes": 6,
        "n_edges": 5,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/beam2.bif"
