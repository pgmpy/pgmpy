from .._base import BaseExampleModel, BIFMixin


class Fingermarks1(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/fingermarks1",
        "n_nodes": 12,
        "n_edges": 20,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/fingermarks1.bif"
