from .._base import BaseExampleModel, BIFMixin


class Accidents(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/accidents",
        "n_nodes": 17,
        "n_edges": 16,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/accidents.bif"
