from .._base import BaseExampleModel, BIFMixin


class Urinary(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/urinary",
        "n_nodes": 36,
        "n_edges": 109,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/urinary.bif"
