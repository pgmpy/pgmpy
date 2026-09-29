from .._base import BaseExampleModel, BIFMixin


class Argument(BIFMixin, BaseExampleModel):
    _tags = {
        "name": "bnrep/argument",
        "n_nodes": 10,
        "n_edges": 11,
        "is_parameterized": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_hybrid": False,
    }

    data_url = "bnrep/argument.bif"
