from .._base import BaseExampleModel, BIFMixin


class Covid3(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/covid3",
        "n_nodes": 12,
        "n_edges": 30,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/covid3.bif"
