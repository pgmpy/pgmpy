from .._base import PlainBIFMixin, _BaseExampleModel


class Corticosteroid(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/corticosteroid",
        "n_nodes": 3,
        "n_edges": 3,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/corticosteroid.bif"
