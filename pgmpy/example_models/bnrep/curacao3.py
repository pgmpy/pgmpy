from .._base import PlainBIFMixin, _BaseExampleModel


class Curacao3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/curacao3",
        "n_nodes": 19,
        "n_edges": 18,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/curacao3.bif"
