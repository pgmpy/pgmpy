from .._base import PlainBIFMixin, _BaseExampleModel


class Coral3(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/coral3",
        "n_nodes": 8,
        "n_edges": 12,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/coral3.bif"
