from .._base import BaseExampleModel, BIFMixin


class Airegulation3(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/airegulation3",
        "n_nodes": 19,
        "n_edges": 37,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/airegulation3.bif"
