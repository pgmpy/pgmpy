from .._base import BaseExampleModel, BIFMixin


class Theft2(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/theft2",
        "n_nodes": 5,
        "n_edges": 4,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/theft2.bif"
