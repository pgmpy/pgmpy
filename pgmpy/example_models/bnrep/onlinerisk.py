from .._base import BaseExampleModel, BIFMixin


class Onlinerisk(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/onlinerisk",
        "n_nodes": 84,
        "n_edges": 109,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/onlinerisk.bif"
