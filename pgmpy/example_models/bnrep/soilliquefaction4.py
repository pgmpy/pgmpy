from .._base import BaseExampleModel, BIFMixin


class Soilliquefaction4(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/soilliquefaction4",
        "n_nodes": 7,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/soilliquefaction4.bif"
