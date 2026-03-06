from .._base import BIFMixin, _BaseExampleModel


class Charleston(BIFMixin, _BaseExampleModel):
    _tags = {
        "name": "bnrep/charleston",
        "n_nodes": 24,
        "n_edges": 35,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/charleston.bif"
