from .._base import BaseExampleModel, BIFMixin


class Kosterhavet(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/kosterhavet",
        "n_nodes": 38,
        "n_edges": 63,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/kosterhavet.bif"
