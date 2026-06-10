from .._base import BaseExampleModel, BIFMixin


class Criminal3(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/criminal3",
        "n_nodes": 8,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/criminal3.bif"
