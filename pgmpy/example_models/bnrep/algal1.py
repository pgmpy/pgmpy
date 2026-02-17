from .._base import PlainBIFMixin, _BaseExampleModel


class Algal1(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/algal1",
        "n_nodes": 9,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/algal1.bif"
