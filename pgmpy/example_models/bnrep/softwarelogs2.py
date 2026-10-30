from .._base import BaseExampleModel, BIFMixin


class Softwarelogs2(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/softwarelogs2",
        "n_nodes": 40,
        "n_edges": 67,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/softwarelogs2.bif"
