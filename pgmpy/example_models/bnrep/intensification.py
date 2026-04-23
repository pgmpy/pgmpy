from .._base import BIFMixin, _BaseExampleModel


class Intensification(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/intensification",
        "n_nodes": 16,
        "n_edges": 29,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/intensification.bif"
