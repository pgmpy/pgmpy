from .._base import BaseExampleModel, BIFMixin


class Soil(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/soil",
        "n_nodes": 6,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/soil.bif"
