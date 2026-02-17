from .._base import PlainBIFMixin, _BaseExampleModel


class Algorithms3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/algorithms3",
        "n_nodes": 4,
        "n_edges": 3,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/algorithms3.bif"
