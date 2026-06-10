from .._base import BaseExampleModel, BIFMixin


class Cardiovascular(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/cardiovascular",
        "n_nodes": 13,
        "n_edges": 40,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/cardiovascular.bif"
