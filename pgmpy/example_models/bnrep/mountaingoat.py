from .._base import PlainBIFMixin, _BaseExampleModel


class Mountaingoat(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/mountaingoat",
        "n_nodes": 7,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/mountaingoat.bif"
