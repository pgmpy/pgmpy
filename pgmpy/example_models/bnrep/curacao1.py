from .._base import BIFMixin, _BaseExampleModel


class Curacao1(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/curacao1",
        "n_nodes": 13,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/curacao1.bif"
