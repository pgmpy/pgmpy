from .._base import BaseExampleModel, BIFMixin


class Soilliquefaction1(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/soilliquefaction1",
        "n_nodes": 7,
        "n_edges": 6,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/soilliquefaction1.bif"
