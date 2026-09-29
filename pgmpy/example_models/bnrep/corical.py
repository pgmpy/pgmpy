from .._base import BaseExampleModel, BIFMixin


class Corical(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/corical",
        "n_nodes": 20,
        "n_edges": 26,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/corical.bif"
