from .._base import BaseExampleModel, BIFMixin


class Agropastoral2(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/agropastoral2",
        "n_nodes": 11,
        "n_edges": 17,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/agropastoral2.bif"
