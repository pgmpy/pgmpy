from .._base import PlainBIFMixin, _BaseExampleModel


class Macrophytes(PlainBIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/macrophytes",
        "n_nodes": 15,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/macrophytes.bif"
