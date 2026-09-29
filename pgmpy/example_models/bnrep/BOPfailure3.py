from .._base import BaseExampleModel, BIFMixin


class BOPfailure3(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/BOPfailure3",
        "n_nodes": 28,
        "n_edges": 27,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/BOPfailure3.bif"
