from .._base import BIFMixin, _BaseExampleModel


class Conasense(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/conasense",
        "n_nodes": 4,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/conasense.bif"
