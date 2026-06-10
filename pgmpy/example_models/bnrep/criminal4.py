from .._base import BaseExampleModel, BIFMixin


class Criminal4(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/criminal4",
        "n_nodes": 8,
        "n_edges": 9,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/criminal4.bif"
